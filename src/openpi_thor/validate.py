from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import jax
import numpy as np

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

from openpi_thor._schema import ArtifactBundle
from openpi_thor._schema import ValidationReport
from openpi_thor.calibration import sample_dataset_examples
from openpi_thor.runtime import load_pytorch_bundle
from openpi_thor.runtime import load_tensorrt_policy

logger = logging.getLogger(__name__)


def _resolve_train_config(config: str | _config.TrainConfig) -> _config.TrainConfig:
    if isinstance(config, _config.TrainConfig):
        return config
    return _config.get_config(config)


def _resolve_bundle(bundle_dir: str | Path, *, config_name: str) -> ArtifactBundle:
    bundle_path = Path(bundle_dir).expanduser().resolve()
    metadata_path = bundle_path / "openpi_thor_bundle.json"
    if metadata_path.exists():
        return ArtifactBundle.load(bundle_path)
    bundle = ArtifactBundle(bundle_dir=bundle_path, config_name=config_name)
    bundle.save()
    return bundle


def _clone_example(example: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy({k: v for k, v in example.items() if k != "dataset_index"})


def _make_noise(action_horizon: int, action_dim: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((action_horizon, action_dim), dtype=np.float32)


def _cosine_similarity(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs_flat = lhs.reshape(-1)
    rhs_flat = rhs.reshape(-1)
    denom = np.linalg.norm(lhs_flat) * np.linalg.norm(rhs_flat)
    if denom == 0:
        return 1.0
    return float(np.dot(lhs_flat, rhs_flat) / denom)


def _default_thresholds(
    bundle: ArtifactBundle,
    candidate_backend: str,
    *,
    reference_backend: str = "jax",
    reference_path: str | Path | None = None,
    candidate_path: str | Path | None = None,
) -> dict[str, float]:
    """Choose default similarity/error thresholds for one validation run."""

    reference_precision = _artifact_precision_from_path(reference_path)
    candidate_precision = _artifact_precision_from_path(candidate_path, fallback=bundle.precision)
    if (
        reference_backend == "jax"
        and candidate_backend == "pytorch"
        and not (reference_precision and reference_precision.startswith("fp8"))
    ):
        return {"min_cosine": 0.999, "mean_abs_error": 0.01, "max_abs_error": 0.05}
    if any(
        precision is not None and precision.startswith("fp8")
        for precision in (reference_precision, candidate_precision)
    ):
        return {"min_cosine": 0.97, "mean_abs_error": 0.08, "max_abs_error": 0.3}
    return {"min_cosine": 0.995, "mean_abs_error": 0.03, "max_abs_error": 0.12}


def _artifact_precision_from_path(path: str | Path | None, *, fallback: str | None = None) -> str | None:
    if path is None:
        return fallback
    stem = Path(path).stem.lower()
    if "nvfp4" in stem:
        return "fp8_nvfp4"
    if "fp8" in stem:
        return "fp8"
    if "fp16" in stem:
        return "fp16"
    return fallback


def _validation_key(
    candidate_backend: str,
    candidate_path: str | None,
    *,
    reference_backend: str = "jax",
    reference_path: str | None = None,
) -> str:
    if candidate_backend == "tensorrt" and reference_backend == "tensorrt" and candidate_path and reference_path:
        return f"tensorrt:{Path(reference_path).stem}:vs:{Path(candidate_path).stem}"
    if candidate_backend != "tensorrt" or candidate_path is None:
        return candidate_backend
    return f"{candidate_backend}:{Path(candidate_path).stem}"


def _resolve_reference_checkpoint_dir(
    bundle: ArtifactBundle,
    reference_checkpoint_dir: str | Path | None,
) -> str | Path:
    if reference_checkpoint_dir is not None:
        return reference_checkpoint_dir
    if bundle.source_checkpoint_dir is None:
        raise ValueError("reference_checkpoint_dir is required when the bundle has no source checkpoint metadata.")
    return bundle.source_checkpoint_dir


def _resolve_engine_path(bundle: ArtifactBundle, engine_path: str | Path | None) -> Path:
    if engine_path is not None:
        return Path(engine_path).expanduser().resolve()
    if recommended := bundle.get_recommended_engine_path():
        return recommended.expanduser().resolve()
    if bundle.engine_paths:
        return Path(next(iter(bundle.engine_paths.values()))).expanduser().resolve()
    raise FileNotFoundError("No TensorRT engine path was provided and the bundle has no recorded engine.")


def _load_policy_for_backend(
    train_config: _config.TrainConfig,
    bundle: ArtifactBundle,
    *,
    backend: str,
    default_prompt: str | None,
    pytorch_device: str | None,
    reference_checkpoint_dir: str | Path | None = None,
    engine_path: str | Path | None = None,
    reference_jax_platform: str = "cpu",
):
    if backend == "jax":
        resolved_checkpoint_dir = _resolve_reference_checkpoint_dir(bundle, reference_checkpoint_dir)
        if reference_jax_platform:
            try:
                jax.config.update("jax_platforms", reference_jax_platform)
            except Exception:  # noqa: BLE001
                pass
        policy = _policy_config.create_trained_policy(
            train_config,
            resolved_checkpoint_dir,
            default_prompt=default_prompt,
        )
        return policy, str(Path(resolved_checkpoint_dir).expanduser().resolve()), None

    if backend == "pytorch":
        policy, _ = load_pytorch_bundle(
            train_config,
            bundle.bundle_dir,
            default_prompt=default_prompt,
            pytorch_device=pytorch_device,
        )
        weight_path = bundle.weight_path.expanduser().resolve()
        return policy, str(weight_path), _artifact_precision_from_path(weight_path, fallback=bundle.precision)

    if backend == "tensorrt":
        resolved_engine_path = _resolve_engine_path(bundle, engine_path)
        policy = load_tensorrt_policy(
            train_config,
            bundle.bundle_dir,
            engine_path=resolved_engine_path,
            require_validated=False,
            default_prompt=default_prompt,
            pytorch_device=pytorch_device,
        )
        return policy, str(resolved_engine_path), _artifact_precision_from_path(
            resolved_engine_path, fallback=bundle.precision
        )

    raise ValueError(f"Unsupported backend={backend!r}")


def _compare_policy_outputs(
    train_config: _config.TrainConfig,
    *,
    examples: list[dict[str, Any]],
    reference_policy,
    candidate_policy,
    reference_backend: str,
    candidate_backend: str,
    config_name: str,
    reference_path: str | None = None,
    reference_precision: str | None = None,
    candidate_path: str | None = None,
    candidate_precision: str | None = None,
    thresholds: dict[str, float],
) -> ValidationReport:
    per_example: list[dict[str, float | int]] = []
    cosine_values: list[float] = []
    mean_abs_values: list[float] = []
    max_abs_values: list[float] = []

    for example_index, example in enumerate(examples):
        noise = _make_noise(train_config.model.action_horizon, train_config.model.action_dim, seed=example_index)
        reference = reference_policy.infer(_clone_example(example), noise=noise)["actions"]
        candidate = candidate_policy.infer(_clone_example(example), noise=noise)["actions"]
        cosine = _cosine_similarity(reference, candidate)
        mean_abs = float(np.mean(np.abs(reference - candidate)))
        max_abs = float(np.max(np.abs(reference - candidate)))
        cosine_values.append(cosine)
        mean_abs_values.append(mean_abs)
        max_abs_values.append(max_abs)
        per_example.append(
            {
                "dataset_index": int(np.asarray(example.get("dataset_index", example_index)).item()),
                "cosine": cosine,
                "mean_abs_error": mean_abs,
                "max_abs_error": max_abs,
            }
        )

    reference_label = reference_backend if reference_path is None else f"{reference_backend} ({Path(reference_path).name})"
    candidate_label = candidate_backend if candidate_path is None else f"{candidate_backend} ({Path(candidate_path).name})"
    return ValidationReport(
        reference_backend=reference_backend,
        candidate_backend=candidate_backend,
        config_name=config_name,
        reference_path=reference_path,
        reference_precision=reference_precision,
        candidate_path=candidate_path,
        precision=candidate_precision,
        num_examples=len(per_example),
        passed=(
            min(cosine_values) >= thresholds["min_cosine"]
            and float(np.mean(mean_abs_values)) <= thresholds["mean_abs_error"]
            and max(max_abs_values) <= thresholds["max_abs_error"]
        ),
        mean_cosine=float(np.mean(cosine_values)),
        min_cosine=min(cosine_values),
        mean_abs_error=float(np.mean(mean_abs_values)),
        max_abs_error=max(max_abs_values),
        thresholds=thresholds,
        per_example=per_example,
        notes=[
            f"Compared {candidate_label} against {reference_label} with fixed-noise inference.",
            "Examples were sampled through the config's LeRobot repack path before policy inference.",
        ],
    )


def _record_validation_report(
    bundle: ArtifactBundle,
    *,
    report: ValidationReport,
    validation_key: str,
    artifact_key: str | None,
    dataset_repo_id: str | None,
    dataset_root: str | Path | None,
) -> ValidationReport:
    bundle.set_validation_report(
        validation_key,
        report,
        artifact_key=artifact_key,
    )
    bundle.write_report(
        f"validate_{validation_key.replace(':', '_')}",
        {
            "phase": "validate",
            "validation_key": validation_key,
            "dataset_repo_id": dataset_repo_id,
            "dataset_root": str(dataset_root) if dataset_root is not None else None,
            "report": report.to_dict(),
        },
        artifact_key=artifact_key,
        report_key=f"validate:{validation_key}",
    )
    return report


def _log_validation_summary(report: ValidationReport) -> None:
    """Emit a concise one-line validation summary."""

    logger.info(
        "Validation summary: passed=%s mean_cos=%.6f min_cos=%.6f mean_abs=%.6f max_abs=%.6f",
        report.passed,
        report.mean_cosine,
        report.min_cosine,
        report.mean_abs_error,
        report.max_abs_error,
    )


def compare_backends(
    config: str | _config.TrainConfig,
    bundle_dir: str | Path,
    *,
    examples: list[dict[str, Any]] | None = None,
    reference_checkpoint_dir: str | Path | None = None,
    candidate_backend: str = "pytorch",
    engine_path: str | Path | None = None,
    num_examples: int = 8,
    thresholds: dict[str, float] | None = None,
    default_prompt: str | None = None,
    pytorch_device: str | None = None,
    reference_jax_platform: str = "cpu",
    dataset_repo_id: str | None = None,
    dataset_root: str | Path | None = None,
) -> ValidationReport:
    """Compare a PyTorch or TensorRT candidate against the JAX reference model."""

    train_config = _resolve_train_config(config)
    bundle = _resolve_bundle(bundle_dir, config_name=train_config.name)
    logger.info("Validating %s against JAX for bundle %s", candidate_backend, bundle.bundle_dir)

    if examples is None:
        examples = sample_dataset_examples(
            train_config,
            num_examples=num_examples,
            dataset_repo_id=dataset_repo_id,
            dataset_root=dataset_root,
        )
    logger.info("Loaded %d validation example(s)", len(examples))

    reference_policy, reference_path, reference_precision = _load_policy_for_backend(
        train_config,
        bundle,
        backend="jax",
        reference_checkpoint_dir=reference_checkpoint_dir,
        default_prompt=default_prompt,
        pytorch_device=pytorch_device,
        reference_jax_platform=reference_jax_platform,
    )
    candidate_policy, candidate_path, candidate_precision = _load_policy_for_backend(
        train_config,
        bundle,
        backend=candidate_backend,
        engine_path=engine_path,
        default_prompt=default_prompt,
        pytorch_device=pytorch_device,
        reference_jax_platform=reference_jax_platform,
    )
    logger.info("Reference backend: jax (%s)", reference_path)
    logger.info("Candidate backend: %s (%s)", candidate_backend, candidate_path)

    thresholds = thresholds or _default_thresholds(
        bundle,
        candidate_backend,
        reference_backend="jax",
        reference_path=reference_path,
        candidate_path=candidate_path,
    )
    report = _compare_policy_outputs(
        train_config,
        examples=examples,
        reference_policy=reference_policy,
        candidate_policy=candidate_policy,
        reference_backend="jax",
        candidate_backend=candidate_backend,
        config_name=train_config.name,
        reference_path=reference_path,
        reference_precision=reference_precision,
        candidate_path=candidate_path,
        candidate_precision=candidate_precision,
        thresholds=thresholds,
    )

    artifact_key = candidate_precision
    validation_key = _validation_key(
        candidate_backend,
        candidate_path,
        reference_backend="jax",
        reference_path=reference_path,
    )
    _record_validation_report(
        bundle,
        report=report,
        validation_key=validation_key,
        artifact_key=artifact_key,
        dataset_repo_id=dataset_repo_id,
        dataset_root=dataset_root,
    )
    if candidate_backend == "tensorrt" and report.passed and candidate_path is not None:
        bundle.set_recommended_engine(candidate_path, artifact_key=artifact_key)
        logger.info("Marked %s as the recommended engine", candidate_path)
    bundle.save()
    _log_validation_summary(report)
    logger.info("Updated bundle manifest %s", bundle.metadata_path)
    return report


def compare_tensorrt_engines(
    config: str | _config.TrainConfig,
    bundle_dir: str | Path,
    *,
    candidate_engine_path: str | Path,
    reference_engine_path: str | Path | None = None,
    examples: list[dict[str, Any]] | None = None,
    num_examples: int = 8,
    thresholds: dict[str, float] | None = None,
    default_prompt: str | None = None,
    pytorch_device: str | None = None,
    dataset_repo_id: str | None = None,
    dataset_root: str | Path | None = None,
) -> ValidationReport:
    """Compare two TensorRT engines directly on the same fixed-noise dataset inputs."""

    train_config = _resolve_train_config(config)
    bundle = _resolve_bundle(bundle_dir, config_name=train_config.name)
    logger.info("Comparing TensorRT engines for bundle %s", bundle.bundle_dir)

    if examples is None:
        examples = sample_dataset_examples(
            train_config,
            num_examples=num_examples,
            dataset_repo_id=dataset_repo_id,
            dataset_root=dataset_root,
        )
    logger.info("Loaded %d validation example(s)", len(examples))

    reference_policy, resolved_reference_path, reference_precision = _load_policy_for_backend(
        train_config,
        bundle,
        backend="tensorrt",
        engine_path=reference_engine_path,
        default_prompt=default_prompt,
        pytorch_device=pytorch_device,
    )
    candidate_policy, resolved_candidate_path, candidate_precision = _load_policy_for_backend(
        train_config,
        bundle,
        backend="tensorrt",
        engine_path=candidate_engine_path,
        default_prompt=default_prompt,
        pytorch_device=pytorch_device,
    )
    logger.info("Reference engine: %s", resolved_reference_path)
    logger.info("Candidate engine: %s", resolved_candidate_path)

    thresholds = thresholds or _default_thresholds(
        bundle,
        "tensorrt",
        reference_backend="tensorrt",
        reference_path=resolved_reference_path,
        candidate_path=resolved_candidate_path,
    )
    report = _compare_policy_outputs(
        train_config,
        examples=examples,
        reference_policy=reference_policy,
        candidate_policy=candidate_policy,
        reference_backend="tensorrt",
        candidate_backend="tensorrt",
        config_name=train_config.name,
        reference_path=resolved_reference_path,
        reference_precision=reference_precision,
        candidate_path=resolved_candidate_path,
        candidate_precision=candidate_precision,
        thresholds=thresholds,
    )

    validation_key = _validation_key(
        "tensorrt",
        resolved_candidate_path,
        reference_backend="tensorrt",
        reference_path=resolved_reference_path,
    )
    _record_validation_report(
        bundle,
        report=report,
        validation_key=validation_key,
        artifact_key=candidate_precision,
        dataset_repo_id=dataset_repo_id,
        dataset_root=dataset_root,
    )
    bundle.save()
    _log_validation_summary(report)
    logger.info("Updated bundle manifest %s", bundle.metadata_path)
    return report
