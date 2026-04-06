import json
from pathlib import Path

from openpi_thor._schema import ArtifactBundle
from openpi_thor._schema import ExportOptions
from openpi_thor._schema import ValidationReport


def test_artifact_records_and_status_round_trip(tmp_path: Path) -> None:
    bundle = ArtifactBundle(bundle_dir=tmp_path, config_name="pi05_xlerobot_pinc_finetune")
    onnx_path = tmp_path / "onnx" / "model_fp16.onnx"
    engine_path = tmp_path / "engine" / "model_fp16_strongly_typed.engine"
    bundle.write_report("convert_jax_to_pytorch", {"phase": "convert_jax_to_pytorch"})

    bundle.set_onnx_path(
        "fp16",
        onnx_path,
        precision="fp16",
        num_steps=10,
        calibration_source=None,
        calibration_num_samples=None,
        export_options={"precision": "fp16", "num_steps": 10},
    )
    bundle.set_engine_path("model_fp16_strongly_typed", engine_path, artifact_key="fp16")
    bundle.set_validation_report(
        "tensorrt:model_fp16_strongly_typed",
        ValidationReport(
            reference_backend="jax",
            candidate_backend="tensorrt",
            config_name=bundle.config_name,
            reference_path="/tmp/checkpoint",
            candidate_path=str(engine_path),
            reference_precision=None,
            precision="fp16",
            passed=True,
            per_example=[{"dataset_index": 1, "cosine": 0.99, "mean_abs_error": 0.1, "max_abs_error": 0.2}],
            notes=["full report details should live in reports/"],
        ),
        artifact_key="fp16",
    )
    bundle.write_report(
        "validate_tensorrt_model_fp16_strongly_typed",
        {
            "phase": "validate",
            "report": {
                "candidate_backend": "tensorrt",
                "per_example": [{"dataset_index": 1}],
            },
        },
        artifact_key="fp16",
        report_key="validate:tensorrt:model_fp16_strongly_typed",
    )
    bundle.set_recommended_engine(engine_path, artifact_key="fp16")
    bundle.save()

    loaded = ArtifactBundle.load(tmp_path)

    assert loaded.recommended_engine == str(engine_path)
    assert loaded.recommended_artifact == "fp16"
    assert loaded.artifacts["fp16"].onnx_path == str(onnx_path)
    assert loaded.artifacts["fp16"].engine_paths["model_fp16_strongly_typed"] == str(engine_path)
    assert loaded.artifacts["fp16"].recommended_engine_path == str(engine_path)
    assert loaded.report_paths["convert_jax_to_pytorch"] == "reports/convert_jax_to_pytorch.json"
    assert (
        loaded.artifacts["fp16"].report_paths["validate:tensorrt:model_fp16_strongly_typed"]
        == "reports/validate_tensorrt_model_fp16_strongly_typed.json"
    )
    status = loaded.status_dict()
    assert status["recommended_engine"] == "engine/model_fp16_strongly_typed.engine"
    assert status["recommended_artifact"] == "fp16"
    assert status["artifacts"]["fp16"]["files"]["onnx"] == "onnx/model_fp16.onnx"
    assert status["bundle_reports"]["convert_jax_to_pytorch"] == "reports/convert_jax_to_pytorch.json"
    assert status["artifacts"]["fp16"]["reports"]["validate:tensorrt:model_fp16_strongly_typed"] == (
        "reports/validate_tensorrt_model_fp16_strongly_typed.json"
    )
    assert status["artifacts"]["fp16"]["validations"]["tensorrt:model_fp16_strongly_typed"]["report_path"] == (
        "reports/validate_tensorrt_model_fp16_strongly_typed.json"
    )

    verbose_status = loaded.status_dict(verbose=True)
    assert verbose_status["bundle_report_payloads"]["convert_jax_to_pytorch"]["phase"] == "convert_jax_to_pytorch"
    assert (
        verbose_status["artifacts"]["fp16"]["report_payloads"]["validate:tensorrt:model_fp16_strongly_typed"]["phase"]
        == "validate"
    )

    manifest = json.loads((tmp_path / "openpi_thor_bundle.json").read_text())
    summary = manifest["artifacts"]["fp16"]["validations"]["tensorrt:model_fp16_strongly_typed"]
    assert "per_example" not in summary
    assert "notes" not in summary
    assert summary["reference_path"] == "/tmp/checkpoint"
    assert summary["report_path"] == "reports/validate_tensorrt_model_fp16_strongly_typed.json"
    assert manifest["recommended_engine"] == "engine/model_fp16_strongly_typed.engine"
    assert manifest["recommended_artifact"] == "fp16"
    assert "engine_paths" not in manifest
    assert "onnx_paths" not in manifest
    assert "validation_reports" not in manifest
    assert "report_paths" not in manifest
    assert manifest["bundle_reports"]["convert_jax_to_pytorch"] == "reports/convert_jax_to_pytorch.json"
    assert manifest["checkpoint_load_summary"] is None


def test_old_manifest_loads_and_saves_in_compact_shape(tmp_path: Path) -> None:
    legacy_manifest = {
        "bundle_dir": str(tmp_path),
        "config_name": "pi05_xlerobot_pinc_finetune",
        "source_checkpoint_dir": "/tmp/checkpoint",
        "precision": "fp16",
        "num_steps": 10,
        "calibration_source": None,
        "calibration_num_samples": None,
        "onnx_paths": {"fp16": str(tmp_path / "onnx" / "model_fp16.onnx")},
        "engine_paths": {"model_fp16": str(tmp_path / "engine" / "model_fp16.engine")},
        "checkpoint_load_report": {
            "total_checkpoint_keys": 10,
            "loaded_keys": 10,
            "unexpected_keys": [],
            "missing_keys": [],
            "shape_mismatches": [],
            "fail_closed": True,
            "clean": True,
        },
        "validation_reports": {
            "tensorrt:model_fp16": {
                "reference_backend": "jax",
                "candidate_backend": "tensorrt",
                "config_name": "pi05_xlerobot_pinc_finetune",
                "reference_path": "/tmp/checkpoint",
                "reference_precision": None,
                "candidate_path": str(tmp_path / "engine" / "model_fp16.engine"),
                "precision": "fp16",
                "num_examples": 4,
                "passed": True,
                "mean_cosine": 0.99,
                "min_cosine": 0.98,
                "mean_abs_error": 0.01,
                "max_abs_error": 0.02,
                "thresholds": {"min_cosine": 0.9, "mean_abs_error": 0.1, "max_abs_error": 0.2},
            }
        },
        "report_paths": {
            "convert_jax_to_pytorch": str(tmp_path / "reports" / "convert_jax_to_pytorch.json"),
            "validate_tensorrt_model_fp16": str(tmp_path / "reports" / "validate_tensorrt_model_fp16.json"),
        },
        "artifacts": {
            "fp16": {
                "key": "fp16",
                "precision": "fp16",
                "num_steps": 10,
                "calibration_source": None,
                "calibration_num_samples": None,
                "onnx_path": str(tmp_path / "onnx" / "model_fp16.onnx"),
                "engine_paths": {"model_fp16": str(tmp_path / "engine" / "model_fp16.engine")},
                "validation_reports": {
                    "tensorrt:model_fp16": {
                        "reference_backend": "jax",
                        "candidate_backend": "tensorrt",
                        "config_name": "pi05_xlerobot_pinc_finetune",
                        "reference_path": "/tmp/checkpoint",
                        "reference_precision": None,
                        "candidate_path": str(tmp_path / "engine" / "model_fp16.engine"),
                        "precision": "fp16",
                        "num_examples": 4,
                        "passed": True,
                        "mean_cosine": 0.99,
                        "min_cosine": 0.98,
                        "mean_abs_error": 0.01,
                        "max_abs_error": 0.02,
                        "thresholds": {"min_cosine": 0.9, "mean_abs_error": 0.1, "max_abs_error": 0.2},
                    }
                },
                "report_paths": {
                    "validate:tensorrt:model_fp16": str(tmp_path / "reports" / "validate_tensorrt_model_fp16.json")
                },
                "recommended_engine_path": str(tmp_path / "engine" / "model_fp16.engine"),
                "export_options": {"precision": "fp16", "num_steps": 10},
                "extra": {},
            }
        },
        "recommended_engine": str(tmp_path / "engine" / "model_fp16.engine"),
        "extra": {},
    }
    metadata_path = tmp_path / "openpi_thor_bundle.json"
    metadata_path.write_text(json.dumps(legacy_manifest, indent=2))

    loaded = ArtifactBundle.load(tmp_path)

    assert loaded.recommended_engine == str((tmp_path / "engine" / "model_fp16.engine").resolve())
    assert loaded.report_paths == {"convert_jax_to_pytorch": "reports/convert_jax_to_pytorch.json"}
    assert "tensorrt:model_fp16" not in loaded.validation_reports
    assert "tensorrt:model_fp16" in loaded.artifacts["fp16"].validation_reports
    assert loaded.artifacts["fp16"].report_paths == {
        "validate:tensorrt:model_fp16": "reports/validate_tensorrt_model_fp16.json"
    }

    loaded.save()
    compact = json.loads(metadata_path.read_text())

    assert "engine_paths" not in compact
    assert "onnx_paths" not in compact
    assert "validation_reports" not in compact
    assert "report_paths" not in compact
    assert compact["recommended_engine"] == "engine/model_fp16.engine"
    assert compact["artifacts"]["fp16"]["files"]["recommended_engine"] == "engine/model_fp16.engine"
    assert compact["bundle_reports"] == {"convert_jax_to_pytorch": "reports/convert_jax_to_pytorch.json"}


def test_export_options_default_calibration_count_is_32() -> None:
    assert ExportOptions().num_calibration_samples == 32
