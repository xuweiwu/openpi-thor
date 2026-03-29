from __future__ import annotations

from pathlib import Path
from typing import Any

from openpi.training import config as _config

from openpi_thor._schema import ArtifactBundle
from openpi_thor._schema import EngineProfile
from openpi_thor._schema import ExportOptions
from openpi_thor.engine import build_engine
from openpi_thor.export import export_to_onnx_bundle
from openpi_thor.validate import compare_backends


def _resolve_train_config(config: str | _config.TrainConfig) -> _config.TrainConfig:
    if isinstance(config, _config.TrainConfig):
        return config
    return _config.get_config(config)


def prepare_engine(
    config: str | _config.TrainConfig,
    bundle_dir: str | Path,
    *,
    export_options: ExportOptions,
    profile: EngineProfile = EngineProfile(),
    validate: bool = False,
    reference_checkpoint_dir: str | Path | None = None,
    validation_num_examples: int = 8,
    default_prompt: str | None = None,
    pytorch_device: str | None = None,
    dataset_repo_id: str | None = None,
    dataset_root: str | Path | None = None,
) -> ArtifactBundle:
    """Run the standard export -> build -> optional validate workflow.

    This is the high-level orchestration entrypoint used by the `prepare-engine`
    CLI command. It keeps all generated artifacts inside one bundle directory.
    """

    train_config = _resolve_train_config(config)
    bundle = export_to_onnx_bundle(
        train_config,
        bundle_dir,
        options=export_options,
        default_prompt=default_prompt,
        pytorch_device=pytorch_device,
        dataset_repo_id=dataset_repo_id,
        dataset_root=dataset_root,
    )
    artifact_key = bundle.precision or export_options.precision.lower()
    onnx_path = bundle.onnx_paths[artifact_key]
    bundle = build_engine(
        train_config,
        bundle.bundle_dir,
        onnx_path=onnx_path,
        profile=profile,
    )

    engine_key = Path(onnx_path).stem
    engine_path = Path(bundle.engine_paths[engine_key])
    if validate:
        report = compare_backends(
            train_config,
            bundle.bundle_dir,
            reference_checkpoint_dir=reference_checkpoint_dir,
            candidate_backend="tensorrt",
            engine_path=engine_path,
            num_examples=validation_num_examples,
            default_prompt=default_prompt,
            pytorch_device=pytorch_device,
            dataset_repo_id=dataset_repo_id,
            dataset_root=dataset_root,
        )
        if report.passed:
            bundle = ArtifactBundle.load(bundle.bundle_dir)
            bundle.set_recommended_engine(engine_path, artifact_key=artifact_key)
            bundle.save()
    elif artifact_key == "fp16" and bundle.get_recommended_engine_path() is None:
        bundle.set_recommended_engine(engine_path, artifact_key=artifact_key)
        bundle.save()

    return bundle


def bundle_status(bundle_dir: str | Path, *, verbose: bool = False) -> dict[str, Any]:
    """Return a human-oriented summary of the bundle manifest and reports."""

    bundle = ArtifactBundle.load(bundle_dir)
    return bundle.status_dict(verbose=verbose)
