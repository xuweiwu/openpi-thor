from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import json
from pathlib import Path
from typing import Any


class OpenPIThorError(RuntimeError):
    """Base error for the Thor deployment helpers."""


class CheckpointLoadError(OpenPIThorError):
    """Raised when a converted PyTorch checkpoint cannot be loaded safely."""


class CalibrationError(OpenPIThorError):
    """Raised when calibration data is missing or invalid."""


class ValidationError(OpenPIThorError):
    """Raised when an artifact is not validated enough for the requested action."""


@dataclasses.dataclass(frozen=True)
class ShapeMismatch:
    """Describe a checkpoint tensor whose shape does not match the live model."""

    key: str
    checkpoint_shape: tuple[int, ...]
    model_shape: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "checkpoint_shape": list(self.checkpoint_shape),
            "model_shape": list(self.model_shape),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ShapeMismatch":
        return cls(
            key=str(data["key"]),
            checkpoint_shape=tuple(int(v) for v in data["checkpoint_shape"]),
            model_shape=tuple(int(v) for v in data["model_shape"]),
        )


@dataclasses.dataclass
class CheckpointLoadReport:
    """Summarize how safely a converted checkpoint loaded into the PyTorch model."""

    total_checkpoint_keys: int = 0
    loaded_keys: int = 0
    unexpected_keys: list[str] = dataclasses.field(default_factory=list)
    missing_keys: list[str] = dataclasses.field(default_factory=list)
    shape_mismatches: list[ShapeMismatch] = dataclasses.field(default_factory=list)
    fail_closed: bool = True
    clean: bool = True

    @property
    def has_issues(self) -> bool:
        return bool(self.unexpected_keys or self.missing_keys or self.shape_mismatches)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_checkpoint_keys": self.total_checkpoint_keys,
            "loaded_keys": self.loaded_keys,
            "unexpected_keys": list(self.unexpected_keys),
            "missing_keys": list(self.missing_keys),
            "shape_mismatches": [item.to_dict() for item in self.shape_mismatches],
            "fail_closed": self.fail_closed,
            "clean": self.clean,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CheckpointLoadReport":
        return cls(
            total_checkpoint_keys=int(data.get("total_checkpoint_keys", 0)),
            loaded_keys=int(data.get("loaded_keys", 0)),
            unexpected_keys=[str(v) for v in data.get("unexpected_keys", [])],
            missing_keys=[str(v) for v in data.get("missing_keys", [])],
            shape_mismatches=[ShapeMismatch.from_dict(v) for v in data.get("shape_mismatches", [])],
            fail_closed=bool(data.get("fail_closed", True)),
            clean=bool(data.get("clean", True)),
        )


@dataclasses.dataclass(frozen=True)
class ExportOptions:
    """User-facing options that shape ONNX export and optional quantization."""

    precision: str = "fp16"
    num_steps: int = 10
    enable_llm_nvfp4: bool = False
    quantize_attention_matmul: bool = False
    num_calibration_samples: int = 32
    allow_dummy_calibration: bool = False


@dataclasses.dataclass(frozen=True)
class EngineProfile:
    """TensorRT build profile options shared across engine-building commands."""

    min_batch: int = 1
    opt_batch: int = 1
    max_batch: int = 1
    min_seq_len: int | None = None
    opt_seq_len: int | None = None
    max_seq_len: int | None = None
    strongly_typed: bool = True
    extra_args: tuple[str, ...] = ()


@dataclasses.dataclass
class ValidationReport:
    """Numerical comparison summary for one backend-versus-JAX validation run."""

    reference_backend: str
    candidate_backend: str
    config_name: str
    reference_path: str | None = None
    reference_precision: str | None = None
    candidate_path: str | None = None
    precision: str | None = None
    num_examples: int = 0
    passed: bool = False
    mean_cosine: float = 0.0
    min_cosine: float = 0.0
    mean_abs_error: float = 0.0
    max_abs_error: float = 0.0
    thresholds: dict[str, float] = dataclasses.field(default_factory=dict)
    per_example: list[dict[str, float | int]] = dataclasses.field(default_factory=list)
    notes: list[str] = dataclasses.field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_backend": self.reference_backend,
            "candidate_backend": self.candidate_backend,
            "config_name": self.config_name,
            "reference_path": self.reference_path,
            "reference_precision": self.reference_precision,
            "candidate_path": self.candidate_path,
            "precision": self.precision,
            "num_examples": self.num_examples,
            "passed": self.passed,
            "mean_cosine": self.mean_cosine,
            "min_cosine": self.min_cosine,
            "mean_abs_error": self.mean_abs_error,
            "max_abs_error": self.max_abs_error,
            "thresholds": dict(self.thresholds),
            "per_example": list(self.per_example),
            "notes": list(self.notes),
        }

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "reference_backend": self.reference_backend,
            "candidate_backend": self.candidate_backend,
            "config_name": self.config_name,
            "reference_path": self.reference_path,
            "reference_precision": self.reference_precision,
            "candidate_path": self.candidate_path,
            "precision": self.precision,
            "num_examples": self.num_examples,
            "passed": self.passed,
            "mean_cosine": self.mean_cosine,
            "min_cosine": self.min_cosine,
            "mean_abs_error": self.mean_abs_error,
            "max_abs_error": self.max_abs_error,
            "thresholds": dict(self.thresholds),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ValidationReport":
        return cls(
            reference_backend=str(data["reference_backend"]),
            candidate_backend=str(data["candidate_backend"]),
            config_name=str(data["config_name"]),
            reference_path=str(data["reference_path"]) if data.get("reference_path") else None,
            reference_precision=str(data["reference_precision"]) if data.get("reference_precision") else None,
            candidate_path=str(data["candidate_path"]) if data.get("candidate_path") else None,
            precision=str(data["precision"]) if data.get("precision") else None,
            num_examples=int(data.get("num_examples", 0)),
            passed=bool(data.get("passed", False)),
            mean_cosine=float(data.get("mean_cosine", 0.0)),
            min_cosine=float(data.get("min_cosine", 0.0)),
            mean_abs_error=float(data.get("mean_abs_error", 0.0)),
            max_abs_error=float(data.get("max_abs_error", 0.0)),
            thresholds={str(k): float(v) for k, v in data.get("thresholds", {}).items()},
            per_example=[dict(item) for item in data.get("per_example", [])],
            notes=[str(v) for v in data.get("notes", [])],
        )


@dataclasses.dataclass
class DoctorReport:
    """Structured output from `openpi-thor doctor`."""

    passed: bool
    errors: list[str] = dataclasses.field(default_factory=list)
    warnings: list[str] = dataclasses.field(default_factory=list)
    info: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "info": dict(self.info),
        }


@dataclasses.dataclass
class ArtifactRecord:
    """Per-artifact state tracked inside a bundle manifest.

    Each precision variant keeps its own ONNX path, engine paths, report files,
    and validation summaries so phase outputs do not overwrite one another.
    """

    key: str
    precision: str
    num_steps: int | None = None
    calibration_source: str | None = None
    calibration_num_samples: int | None = None
    onnx_path: str | None = None
    engine_paths: dict[str, str] = dataclasses.field(default_factory=dict)
    validation_reports: dict[str, ValidationReport] = dataclasses.field(default_factory=dict)
    report_paths: dict[str, str] = dataclasses.field(default_factory=dict)
    recommended_engine_path: str | None = None
    export_options: dict[str, Any] = dataclasses.field(default_factory=dict)
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "precision": self.precision,
            "num_steps": self.num_steps,
            "calibration_source": self.calibration_source,
            "calibration_num_samples": self.calibration_num_samples,
            "onnx_path": self.onnx_path,
            "engine_paths": dict(self.engine_paths),
            "validation_reports": {k: v.to_manifest_dict() for k, v in self.validation_reports.items()},
            "report_paths": dict(self.report_paths),
            "recommended_engine_path": self.recommended_engine_path,
            "export_options": dict(self.export_options),
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArtifactRecord":
        return cls(
            key=str(data["key"]),
            precision=str(data["precision"]),
            num_steps=int(data["num_steps"]) if data.get("num_steps") is not None else None,
            calibration_source=str(data["calibration_source"]) if data.get("calibration_source") else None,
            calibration_num_samples=(
                int(data["calibration_num_samples"]) if data.get("calibration_num_samples") is not None else None
            ),
            onnx_path=str(data["onnx_path"]) if data.get("onnx_path") else None,
            engine_paths={str(k): str(v) for k, v in data.get("engine_paths", {}).items()},
            validation_reports={
                str(k): ValidationReport.from_dict(v) for k, v in data.get("validation_reports", {}).items()
            },
            report_paths={str(k): str(v) for k, v in data.get("report_paths", {}).items()},
            recommended_engine_path=(
                str(data["recommended_engine_path"]) if data.get("recommended_engine_path") else None
            ),
            export_options=dict(data.get("export_options", {})),
            extra=dict(data.get("extra", {})),
        )


@dataclasses.dataclass
class ArtifactBundle:
    """Thin manifest for one model bundle and its derived deployment artifacts."""

    bundle_dir: Path
    config_name: str
    source_checkpoint_dir: str | None = None
    precision: str | None = None
    num_steps: int | None = None
    calibration_source: str | None = None
    calibration_num_samples: int | None = None
    onnx_paths: dict[str, str] = dataclasses.field(default_factory=dict)
    engine_paths: dict[str, str] = dataclasses.field(default_factory=dict)
    checkpoint_load_report: CheckpointLoadReport | None = None
    validation_reports: dict[str, ValidationReport] = dataclasses.field(default_factory=dict)
    report_paths: dict[str, str] = dataclasses.field(default_factory=dict)
    artifacts: dict[str, ArtifactRecord] = dataclasses.field(default_factory=dict)
    recommended_engine: str | None = None
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def metadata_path(self) -> Path:
        return self.bundle_dir / "openpi_thor_bundle.json"

    @property
    def weight_path(self) -> Path:
        return self.bundle_dir / "model.safetensors"

    @property
    def assets_dir(self) -> Path:
        return self.bundle_dir / "assets"

    @property
    def report_dir(self) -> Path:
        return self.bundle_dir / "reports"

    def ensure_artifact(self, key: str, *, precision: str | None = None) -> ArtifactRecord:
        """Create or return the per-artifact record for a precision variant."""

        artifact = self.artifacts.get(key)
        if artifact is None:
            artifact = ArtifactRecord(key=key, precision=precision or key)
            self.artifacts[key] = artifact
        elif precision is not None and artifact.precision != precision:
            artifact.precision = precision
        return artifact

    def set_onnx_path(
        self,
        key: str,
        path: Path,
        *,
        precision: str | None = None,
        num_steps: int | None = None,
        calibration_source: str | None = None,
        calibration_num_samples: int | None = None,
        export_options: Mapping[str, Any] | None = None,
    ) -> None:
        """Record the ONNX path and export metadata for an artifact variant."""

        self.onnx_paths[key] = str(path)
        artifact = self.ensure_artifact(key, precision=precision or key)
        artifact.onnx_path = str(path)
        if num_steps is not None:
            artifact.num_steps = num_steps
        if calibration_source is not None:
            artifact.calibration_source = calibration_source
        if calibration_num_samples is not None:
            artifact.calibration_num_samples = calibration_num_samples
        if export_options is not None:
            artifact.export_options = dict(export_options)

    def set_engine_path(
        self,
        key: str,
        path: Path,
        *,
        artifact_key: str | None = None,
        recommended: bool = False,
    ) -> None:
        """Record a built TensorRT engine and optionally mark it as recommended."""

        self.engine_paths[key] = str(path)
        if artifact_key is not None:
            artifact = self.ensure_artifact(artifact_key, precision=artifact_key)
            artifact.engine_paths[key] = str(path)
            if recommended:
                artifact.recommended_engine_path = str(path)
        if recommended:
            self.recommended_engine = str(path)

    def set_validation_report(
        self,
        key: str,
        report: ValidationReport,
        *,
        artifact_key: str | None = None,
    ) -> None:
        """Attach a validation report to the bundle and, optionally, one artifact variant."""

        self.validation_reports[key] = report
        if report.candidate_backend == "tensorrt" and report.candidate_path:
            engine_key = Path(report.candidate_path).stem
            self.engine_paths[engine_key] = report.candidate_path
        if artifact_key is not None:
            artifact = self.ensure_artifact(artifact_key, precision=artifact_key)
            artifact.validation_reports[key] = report
            if report.candidate_backend == "tensorrt" and report.candidate_path:
                artifact.engine_paths[Path(report.candidate_path).stem] = report.candidate_path

    def set_recommended_engine(self, path: str | Path | None, *, artifact_key: str | None = None) -> None:
        """Mark one engine as the default serving target for this bundle."""

        self.recommended_engine = str(path) if path is not None else None
        if artifact_key is not None:
            artifact = self.ensure_artifact(artifact_key, precision=artifact_key)
            artifact.recommended_engine_path = str(path) if path is not None else None

    def write_report(
        self,
        name: str,
        payload: Mapping[str, Any],
        *,
        artifact_key: str | None = None,
        report_key: str | None = None,
    ) -> Path:
        """Write a detailed JSON phase report under reports/ and record its path."""

        self.report_dir.mkdir(parents=True, exist_ok=True)
        filename = _sanitize_report_name(name)
        path = self.report_dir / filename
        path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True))
        path_ref = _path_ref_for_bundle(self.bundle_dir, path)
        self.report_paths[name] = path_ref
        if artifact_key is not None:
            artifact = self.ensure_artifact(artifact_key, precision=artifact_key)
            artifact.report_paths[report_key or name] = path_ref
        return path

    def resolve_report_path(self, path_ref: str) -> Path:
        path = Path(path_ref)
        if path.is_absolute():
            return path
        return self.bundle_dir / path

    def get_recommended_engine_path(self) -> Path | None:
        if self.recommended_engine:
            return Path(self.recommended_engine)
        for artifact in self.artifacts.values():
            if artifact.recommended_engine_path:
                return Path(artifact.recommended_engine_path)
        return None

    def status_dict(self, *, verbose: bool = False) -> dict[str, Any]:
        """Return a human-oriented summary of the bundle and per-artifact state."""

        artifacts: dict[str, Any] = {}
        keys = set(self.artifacts) | set(self.onnx_paths)
        for key in sorted(keys):
            artifact = self.artifacts.get(key)
            artifacts[key] = {
                "precision": artifact.precision if artifact else key,
                "num_steps": artifact.num_steps if artifact else (self.num_steps if self.precision == key else None),
                "calibration_source": (
                    artifact.calibration_source
                    if artifact
                    else (self.calibration_source if self.precision == key else None)
                ),
                "calibration_num_samples": (
                    artifact.calibration_num_samples
                    if artifact
                    else (self.calibration_num_samples if self.precision == key else None)
                ),
                "onnx_path": artifact.onnx_path if artifact else self.onnx_paths.get(key),
                "engine_paths": dict(artifact.engine_paths) if artifact else {},
                "validation_reports": {
                    report_key: {
                        "passed": report.passed,
                        "candidate_path": report.candidate_path,
                        "mean_cosine": report.mean_cosine,
                        "mean_abs_error": report.mean_abs_error,
                        "max_abs_error": report.max_abs_error,
                    }
                    for report_key, report in (artifact.validation_reports.items() if artifact else ())
                },
                "report_paths": dict(artifact.report_paths) if artifact else {},
                "recommended_engine_path": artifact.recommended_engine_path if artifact else None,
            }
            if verbose and artifact:
                artifacts[key]["reports"] = {
                    report_key: _load_json_if_exists(self.resolve_report_path(path_ref))
                    for report_key, path_ref in artifact.report_paths.items()
                }
        return {
            "bundle_dir": str(self.bundle_dir),
            "config_name": self.config_name,
            "source_checkpoint_dir": self.source_checkpoint_dir,
            "weight_path": str(self.weight_path) if self.weight_path.exists() else None,
            "checkpoint_load_report": self.checkpoint_load_report.to_dict() if self.checkpoint_load_report else None,
            "report_paths": dict(self.report_paths),
            "recommended_engine": self.recommended_engine,
            "artifacts": artifacts,
            **({"reports": {
                report_key: _load_json_if_exists(self.resolve_report_path(path_ref))
                for report_key, path_ref in self.report_paths.items()
            }} if verbose else {}),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle_dir": str(self.bundle_dir),
            "config_name": self.config_name,
            "source_checkpoint_dir": self.source_checkpoint_dir,
            "precision": self.precision,
            "num_steps": self.num_steps,
            "calibration_source": self.calibration_source,
            "calibration_num_samples": self.calibration_num_samples,
            "onnx_paths": dict(self.onnx_paths),
            "engine_paths": dict(self.engine_paths),
            "checkpoint_load_report": self.checkpoint_load_report.to_dict() if self.checkpoint_load_report else None,
            "validation_reports": {k: v.to_manifest_dict() for k, v in self.validation_reports.items()},
            "report_paths": dict(self.report_paths),
            "artifacts": {k: v.to_dict() for k, v in self.artifacts.items()},
            "recommended_engine": self.recommended_engine,
            "extra": dict(self.extra),
        }

    def save(self) -> "ArtifactBundle":
        """Persist the bundle manifest to `openpi_thor_bundle.json`."""

        self.bundle_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.metadata_path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        except PermissionError as exc:
            self.extra["save_error"] = str(exc)
        return self

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArtifactBundle":
        bundle = cls(
            bundle_dir=Path(str(data["bundle_dir"])),
            config_name=str(data["config_name"]),
            source_checkpoint_dir=str(data["source_checkpoint_dir"]) if data.get("source_checkpoint_dir") else None,
            precision=str(data["precision"]) if data.get("precision") else None,
            num_steps=int(data["num_steps"]) if data.get("num_steps") is not None else None,
            calibration_source=str(data["calibration_source"]) if data.get("calibration_source") else None,
            calibration_num_samples=(
                int(data["calibration_num_samples"]) if data.get("calibration_num_samples") is not None else None
            ),
            onnx_paths={str(k): str(v) for k, v in data.get("onnx_paths", {}).items()},
            engine_paths={str(k): str(v) for k, v in data.get("engine_paths", {}).items()},
            checkpoint_load_report=(
                CheckpointLoadReport.from_dict(data["checkpoint_load_report"])
                if data.get("checkpoint_load_report")
                else None
            ),
            validation_reports={
                str(k): ValidationReport.from_dict(v) for k, v in data.get("validation_reports", {}).items()
            },
            report_paths={str(k): str(v) for k, v in data.get("report_paths", {}).items()},
            artifacts={str(k): ArtifactRecord.from_dict(v) for k, v in data.get("artifacts", {}).items()},
            recommended_engine=str(data["recommended_engine"]) if data.get("recommended_engine") else None,
            extra=dict(data.get("extra", {})),
        )
        if not bundle.artifacts:
            _migrate_legacy_bundle_artifacts(bundle)
        return bundle

    @classmethod
    def load(cls, bundle_dir: str | Path) -> "ArtifactBundle":
        """Load a bundle manifest from disk and rebind it to the requested path."""

        bundle_path = Path(bundle_dir).expanduser().resolve()
        metadata_path = bundle_path / "openpi_thor_bundle.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Bundle metadata not found at {metadata_path}")
        bundle = cls.from_dict(json.loads(metadata_path.read_text()))
        bundle.bundle_dir = bundle_path
        return bundle


def default_bundle(bundle_dir: str | Path, config_name: str) -> ArtifactBundle:
    """Create a new in-memory bundle manifest for a config and directory."""

    return ArtifactBundle(bundle_dir=Path(bundle_dir), config_name=config_name)


def _sanitize_report_name(name: str) -> str:
    sanitized = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in name)
    if not sanitized.endswith(".json"):
        sanitized = f"{sanitized}.json"
    return sanitized


def _path_ref_for_bundle(bundle_dir: Path, path: Path) -> str:
    try:
        return str(path.relative_to(bundle_dir))
    except ValueError:
        return str(path)


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _legacy_precision_from_name(name: str, fallback: str | None = None) -> str | None:
    lowered = name.lower()
    if "nvfp4" in lowered:
        return "fp8_nvfp4"
    if "fp8" in lowered:
        return "fp8"
    if "fp16" in lowered:
        return "fp16"
    return fallback


def _migrate_legacy_bundle_artifacts(bundle: ArtifactBundle) -> None:
    """Populate per-artifact records when loading an older single-manifest bundle."""

    for key, onnx_path in bundle.onnx_paths.items():
        calibration_source = None
        calibration_num_samples = None
        if key.startswith("fp8") and (bundle.precision or "").startswith("fp8"):
            calibration_source = bundle.calibration_source
            calibration_num_samples = bundle.calibration_num_samples
        bundle.set_onnx_path(
            key,
            Path(onnx_path),
            precision=key,
            num_steps=bundle.num_steps if bundle.precision == key else None,
            calibration_source=calibration_source,
            calibration_num_samples=calibration_num_samples,
        )
    for engine_key, engine_path in bundle.engine_paths.items():
        artifact_key = _legacy_precision_from_name(engine_key, fallback=bundle.precision) or engine_key
        bundle.set_engine_path(engine_key, Path(engine_path), artifact_key=artifact_key)
    for report_key, report in bundle.validation_reports.items():
        artifact_key = _legacy_precision_from_name(report_key, fallback=report.precision or bundle.precision)
        bundle.set_validation_report(report_key, report, artifact_key=artifact_key)
    if bundle.calibration_source and not (bundle.precision or "").startswith("fp8"):
        fp8_artifacts = [key for key in bundle.artifacts if key.startswith("fp8")]
        if len(fp8_artifacts) == 1:
            artifact = bundle.ensure_artifact(fp8_artifacts[0], precision=fp8_artifacts[0])
            if artifact.calibration_source is None:
                artifact.calibration_source = bundle.calibration_source
            if artifact.calibration_num_samples is None:
                artifact.calibration_num_samples = bundle.calibration_num_samples
    if bundle.recommended_engine is None and bundle.extra.get("recommended_engine"):
        bundle.recommended_engine = str(bundle.extra["recommended_engine"])
        artifact_key = _legacy_precision_from_name(bundle.recommended_engine, fallback=bundle.precision)
        if artifact_key is not None:
            bundle.set_recommended_engine(bundle.recommended_engine, artifact_key=artifact_key)
