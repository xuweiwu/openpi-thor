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
    assert loaded.artifacts["fp16"].onnx_path == str(onnx_path)
    assert loaded.artifacts["fp16"].engine_paths["model_fp16_strongly_typed"] == str(engine_path)
    assert loaded.artifacts["fp16"].recommended_engine_path == str(engine_path)
    assert loaded.report_paths["convert_jax_to_pytorch"] == "reports/convert_jax_to_pytorch.json"
    assert (
        loaded.artifacts["fp16"].report_paths["validate:tensorrt:model_fp16_strongly_typed"]
        == "reports/validate_tensorrt_model_fp16_strongly_typed.json"
    )
    status = loaded.status_dict()
    assert status["recommended_engine"] == str(engine_path)
    assert status["artifacts"]["fp16"]["onnx_path"] == str(onnx_path)
    assert status["report_paths"]["convert_jax_to_pytorch"] == "reports/convert_jax_to_pytorch.json"
    assert status["artifacts"]["fp16"]["report_paths"]["validate:tensorrt:model_fp16_strongly_typed"] == (
        "reports/validate_tensorrt_model_fp16_strongly_typed.json"
    )

    verbose_status = loaded.status_dict(verbose=True)
    assert verbose_status["reports"]["convert_jax_to_pytorch"]["phase"] == "convert_jax_to_pytorch"
    assert (
        verbose_status["artifacts"]["fp16"]["reports"]["validate:tensorrt:model_fp16_strongly_typed"]["phase"]
        == "validate"
    )

    manifest = json.loads((tmp_path / "openpi_thor_bundle.json").read_text())
    summary = manifest["artifacts"]["fp16"]["validation_reports"]["tensorrt:model_fp16_strongly_typed"]
    assert "per_example" not in summary
    assert "notes" not in summary
    assert summary["reference_path"] == "/tmp/checkpoint"


def test_export_options_default_calibration_count_is_32() -> None:
    assert ExportOptions().num_calibration_samples == 32
