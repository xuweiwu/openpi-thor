from __future__ import annotations

import copy
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
    candidate_path: str | Path | None = None,
) -> dict[str, float]:
    """Choose default similarity/error thresholds for a validation run."""

    effective_precision = _artifact_precision_from_path(candidate_path, fallback=bundle.precision)
    if candidate_backend == "pytorch":
        return {"min_cosine": 0.999, "mean_abs_error": 0.01, "max_abs_error": 0.05}
    if effective_precision and effective_precision.startswith("fp8"):
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


def _validation_key(candidate_backend: str, candidate_path: str | None) -> str:
    if candidate_backend != "tensorrt" or candidate_path is None:
        return candidate_backend
    return f"{candidate_backend}:{Path(candidate_path).stem}"


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
    """Compare a PyTorch or TensorRT candidate against the JAX reference model.

    The comparison uses fixed-noise inference on sampled dataset examples and
    records the resulting report back into the bundle manifest and reports/.
    """

    train_config = _resolve_train_config(config)
    bundle = _resolve_bundle(bundle_dir, config_name=train_config.name)

    if reference_checkpoint_dir is None:
        if bundle.source_checkpoint_dir is None:
            raise ValueError("reference_checkpoint_dir is required when the bundle has no source checkpoint metadata.")
        reference_checkpoint_dir = bundle.source_checkpoint_dir

    if examples is None:
        examples = sample_dataset_examples(
            train_config,
            num_examples=num_examples,
            dataset_repo_id=dataset_repo_id,
            dataset_root=dataset_root,
        )

    if reference_jax_platform:
        try:
            jax.config.update("jax_platforms", reference_jax_platform)
        except Exception:  # noqa: BLE001
            pass

    reference_policy = _policy_config.create_trained_policy(
        train_config,
        reference_checkpoint_dir,
        default_prompt=default_prompt,
    )
    if candidate_backend == "pytorch":
        candidate_policy, _ = load_pytorch_bundle(
            train_config,
            bundle.bundle_dir,
            default_prompt=default_prompt,
            pytorch_device=pytorch_device,
        )
        candidate_path = str(bundle.weight_path)
    elif candidate_backend == "tensorrt":
        resolved_engine_path = Path(engine_path).expanduser().resolve() if engine_path is not None else None
        candidate_policy = load_tensorrt_policy(
            train_config,
            bundle.bundle_dir,
            engine_path=resolved_engine_path,
            require_validated=False,
            default_prompt=default_prompt,
            pytorch_device=pytorch_device,
        )
        candidate_path = str(resolved_engine_path or Path(next(iter(bundle.engine_paths.values()))).expanduser().resolve())
    else:
        raise ValueError(f"Unsupported candidate_backend={candidate_backend!r}")

    thresholds = thresholds or _default_thresholds(bundle, candidate_backend, candidate_path=candidate_path)

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

    report = ValidationReport(
        reference_backend="jax",
        candidate_backend=candidate_backend,
        config_name=train_config.name,
        candidate_path=candidate_path,
        precision=_artifact_precision_from_path(candidate_path, fallback=bundle.precision),
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
            f"Compared {candidate_backend} against the JAX checkpoint with fixed-noise inference.",
            "Examples were sampled through the config's LeRobot repack path before policy inference.",
        ],
    )
    artifact_key = _artifact_precision_from_path(candidate_path, fallback=bundle.precision) if candidate_path else None
    validation_key = _validation_key(candidate_backend, candidate_path)
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
    if candidate_backend == "tensorrt" and report.passed and candidate_path is not None:
        bundle.set_recommended_engine(candidate_path, artifact_key=artifact_key)
    bundle.save()
    return report
