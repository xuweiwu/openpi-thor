from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from openpi.models import model as _model
from openpi.training import config as _config

from openpi_thor._schema import ArtifactBundle
from openpi_thor._schema import EngineProfile


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


def _candidate_onnx_path(bundle: ArtifactBundle, onnx_path: str | Path | None) -> Path:
    """Resolve which ONNX artifact should be used for engine building."""

    if onnx_path is not None:
        return Path(onnx_path).expanduser().resolve()
    if bundle.precision and bundle.precision in bundle.onnx_paths:
        return Path(bundle.onnx_paths[bundle.precision]).expanduser().resolve()
    if bundle.onnx_paths:
        return Path(next(iter(bundle.onnx_paths.values()))).expanduser().resolve()
    raise FileNotFoundError("No ONNX artifact found in the bundle.")


def _onnx_input_names(onnx_path: Path) -> set[str]:
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    return {value.name for value in model.graph.input}


def _shape_profiles(train_config: _config.TrainConfig, profile: EngineProfile) -> dict[str, tuple[str, str, str]]:
    """Build TensorRT dynamic-shape profile strings for the exported ONNX inputs."""

    seq = (
        profile.min_seq_len or train_config.model.max_token_len,
        profile.opt_seq_len or train_config.model.max_token_len,
        profile.max_seq_len or train_config.model.max_token_len,
    )
    batch = (profile.min_batch, profile.opt_batch, profile.max_batch)
    images_shape = (
        f"{batch[0]}x{len(_model.IMAGE_KEYS) * 3}x{_model.IMAGE_RESOLUTION[0]}x{_model.IMAGE_RESOLUTION[1]}",
        f"{batch[1]}x{len(_model.IMAGE_KEYS) * 3}x{_model.IMAGE_RESOLUTION[0]}x{_model.IMAGE_RESOLUTION[1]}",
        f"{batch[2]}x{len(_model.IMAGE_KEYS) * 3}x{_model.IMAGE_RESOLUTION[0]}x{_model.IMAGE_RESOLUTION[1]}",
    )
    return {
        "images": images_shape,
        "img_masks": (f"{batch[0]}x{len(_model.IMAGE_KEYS)}", f"{batch[1]}x{len(_model.IMAGE_KEYS)}", f"{batch[2]}x{len(_model.IMAGE_KEYS)}"),
        "lang_tokens": (f"{batch[0]}x{seq[0]}", f"{batch[1]}x{seq[1]}", f"{batch[2]}x{seq[2]}"),
        "lang_masks": (f"{batch[0]}x{seq[0]}", f"{batch[1]}x{seq[1]}", f"{batch[2]}x{seq[2]}"),
        "state": (
            f"{batch[0]}x{train_config.model.action_dim}",
            f"{batch[1]}x{train_config.model.action_dim}",
            f"{batch[2]}x{train_config.model.action_dim}",
        ),
        "noise": (
            f"{batch[0]}x{train_config.model.action_horizon}x{train_config.model.action_dim}",
            f"{batch[1]}x{train_config.model.action_horizon}x{train_config.model.action_dim}",
            f"{batch[2]}x{train_config.model.action_horizon}x{train_config.model.action_dim}",
        ),
    }


def _shape_flag(flag: str, shapes: dict[str, tuple[str, str, str]], index: int, input_names: set[str]) -> str:
    values = [f"{name}:{shape[index]}" for name, shape in shapes.items() if name in input_names]
    return f"--{flag}Shapes=" + ",".join(values)


def _artifact_precision_from_path(path: Path, *, fallback: str | None = None) -> str | None:
    stem = path.stem.lower()
    if "nvfp4" in stem:
        return "fp8_nvfp4"
    if "fp8" in stem:
        return "fp8"
    if "fp16" in stem:
        return "fp16"
    return fallback


def _build_trtexec_command(
    train_config: _config.TrainConfig,
    bundle: ArtifactBundle,
    onnx_path: Path,
    engine_path: Path,
    profile: EngineProfile,
) -> list[str]:
    """Construct the `trtexec` command used to build the TensorRT engine."""

    input_names = _onnx_input_names(onnx_path)
    shapes = _shape_profiles(train_config, profile)
    command = [
        shutil.which("trtexec") or "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--skipInference",
        "--builderOptimizationLevel=5",
        _shape_flag("min", shapes, 0, input_names),
        _shape_flag("opt", shapes, 1, input_names),
        _shape_flag("max", shapes, 2, input_names),
    ]
    effective_precision = _artifact_precision_from_path(onnx_path, fallback=bundle.precision)
    if profile.strongly_typed:
        command.append("--stronglyTyped")
    else:
        command.append("--fp16")
        if effective_precision and effective_precision.startswith("fp8"):
            command.append("--fp8")
    command.extend(profile.extra_args)
    return command


def build_engine(
    config: str | _config.TrainConfig,
    bundle_dir: str | Path,
    onnx_path: str | Path | None = None,
    *,
    profile: EngineProfile = EngineProfile(),
    dry_run: bool = False,
) -> ArtifactBundle:
    """Build a TensorRT engine from a bundle ONNX artifact and record the result."""

    train_config = _resolve_train_config(config)
    bundle = _resolve_bundle(bundle_dir, config_name=train_config.name)
    resolved_onnx_path = _candidate_onnx_path(bundle, onnx_path)
    engine_dir = bundle.bundle_dir / "engine"
    engine_dir.mkdir(parents=True, exist_ok=True)
    engine_path = engine_dir / f"{resolved_onnx_path.stem}.engine"
    command = _build_trtexec_command(train_config, bundle, resolved_onnx_path, engine_path, profile)
    if not dry_run:
        if shutil.which("trtexec") is None:
            raise FileNotFoundError("trtexec was not found on PATH.")
        subprocess.run(command, check=True)
    artifact_key = _artifact_precision_from_path(resolved_onnx_path, fallback=bundle.precision) or resolved_onnx_path.stem
    bundle.set_engine_path(resolved_onnx_path.stem, engine_path, artifact_key=artifact_key)
    bundle.write_report(
        f"build_{resolved_onnx_path.stem}",
        {
            "phase": "build_tensorrt_engine",
            "artifact_key": artifact_key,
            "config_name": train_config.name,
            "onnx_path": str(resolved_onnx_path),
            "engine_path": str(engine_path),
            "dry_run": dry_run,
            "profile": {
                "min_batch": profile.min_batch,
                "opt_batch": profile.opt_batch,
                "max_batch": profile.max_batch,
                "min_seq_len": profile.min_seq_len,
                "opt_seq_len": profile.opt_seq_len,
                "max_seq_len": profile.max_seq_len,
                "strongly_typed": profile.strongly_typed,
                "extra_args": list(profile.extra_args),
            },
            "trtexec_command": command,
        },
        artifact_key=artifact_key,
        report_key=f"build:{resolved_onnx_path.stem}",
    )
    bundle.save()
    return bundle
