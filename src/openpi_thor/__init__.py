from __future__ import annotations

from importlib import import_module

__all__ = [
    "ArtifactBundle",
    "CalibrationError",
    "CheckpointLoadError",
    "CheckpointLoadReport",
    "DoctorReport",
    "DummyCalibrationSource",
    "EngineProfile",
    "ExportOptions",
    "IterableCalibrationSource",
    "LeRobotPi05CalibrationSource",
    "OpenPIThorError",
    "ValidationError",
    "ValidationReport",
    "build_engine",
    "compare_backends",
    "compare_tensorrt_engines",
    "convert_jax_checkpoint",
    "export_to_onnx_bundle",
    "inspect_jax_checkpoint",
    "load_pytorch_bundle",
    "load_tensorrt_policy",
    "prepare_engine",
    "prepare_runtime",
    "bundle_status",
    "run_doctor",
    "sample_dataset_examples",
    "serve",
]

_EXPORTS = {
    "ArtifactBundle": ("openpi_thor._schema", "ArtifactBundle"),
    "CalibrationError": ("openpi_thor._schema", "CalibrationError"),
    "CheckpointLoadError": ("openpi_thor._schema", "CheckpointLoadError"),
    "CheckpointLoadReport": ("openpi_thor._schema", "CheckpointLoadReport"),
    "DoctorReport": ("openpi_thor._schema", "DoctorReport"),
    "EngineProfile": ("openpi_thor._schema", "EngineProfile"),
    "ExportOptions": ("openpi_thor._schema", "ExportOptions"),
    "OpenPIThorError": ("openpi_thor._schema", "OpenPIThorError"),
    "ValidationError": ("openpi_thor._schema", "ValidationError"),
    "ValidationReport": ("openpi_thor._schema", "ValidationReport"),
    "DummyCalibrationSource": ("openpi_thor.calibration", "DummyCalibrationSource"),
    "IterableCalibrationSource": ("openpi_thor.calibration", "IterableCalibrationSource"),
    "LeRobotPi05CalibrationSource": ("openpi_thor.calibration", "LeRobotPi05CalibrationSource"),
    "prepare_runtime": ("openpi_thor.compat", "prepare_runtime"),
    "convert_jax_checkpoint": ("openpi_thor.convert", "convert_jax_checkpoint"),
    "inspect_jax_checkpoint": ("openpi_thor.convert", "inspect_jax_checkpoint"),
    "run_doctor": ("openpi_thor.doctor", "run_doctor"),
    "build_engine": ("openpi_thor.engine", "build_engine"),
    "export_to_onnx_bundle": ("openpi_thor.export", "export_to_onnx_bundle"),
    "load_pytorch_bundle": ("openpi_thor.runtime", "load_pytorch_bundle"),
    "load_tensorrt_policy": ("openpi_thor.runtime", "load_tensorrt_policy"),
    "prepare_engine": ("openpi_thor.workflow", "prepare_engine"),
    "bundle_status": ("openpi_thor.workflow", "bundle_status"),
    "serve": ("openpi_thor.server", "serve"),
    "compare_backends": ("openpi_thor.validate", "compare_backends"),
    "compare_tensorrt_engines": ("openpi_thor.validate", "compare_tensorrt_engines"),
    "sample_dataset_examples": ("openpi_thor.validate", "sample_dataset_examples"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
