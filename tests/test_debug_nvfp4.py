from pathlib import Path

from openpi_thor._schema import ArtifactBundle
from openpi_thor.debug_nvfp4 import _acceptance_summary
from openpi_thor.debug_nvfp4 import _parse_trtexec_stdout
from openpi_thor.debug_nvfp4 import _resolve_existing_candidate_bundle_state
from openpi_thor.debug_nvfp4 import _resolve_existing_artifact_paths
from openpi_thor.debug_nvfp4 import _summarize_trtexec_profile_rows


def test_parse_trtexec_stdout_extracts_gpu_compute_and_throughput() -> None:
    stdout = """
    Throughput: 11.74 qps
    GPU Compute Time: min = 83.12 ms, max = 90.34 ms, mean = 85.09 ms, median = 84.91 ms, percentile(99%) = 90.00 ms
    """

    parsed = _parse_trtexec_stdout(stdout)

    assert parsed["mean_gpu_compute_ms"] == 85.09
    assert parsed["throughput_qps"] == 11.74


def test_summarize_trtexec_profile_rows_detects_cast_dominated_profiles() -> None:
    rows = [
        {"name": "ReplCastMulCast_0", "averageMs": 40.0, "medianMs": 40.0, "percentage": 40.0},
        {"name": "_gemm_mha_v2_0", "averageMs": 10.0, "medianMs": 10.0, "percentage": 10.0},
        {"name": "other", "averageMs": 50.0, "medianMs": 50.0, "percentage": 50.0},
    ]

    summary = _summarize_trtexec_profile_rows(rows)

    assert summary["repl_cast_mul_cast_count"] == 1
    assert summary["fused_mha_count"] == 1
    assert summary["cast_dominated"] is True


def test_acceptance_summary_uses_fp8_relative_error_ratio() -> None:
    acceptance = _acceptance_summary(
        candidate_jax_report={"mean_abs_error": 1.05, "max_abs_error": 2.10},
        baseline_jax_report={"mean_abs_error": 1.0, "max_abs_error": 2.0},
        candidate_profile={"mean_gpu_compute_ms": 90.0, "cast_dominated": False},
        baseline_profile={"mean_gpu_compute_ms": 100.0, "cast_dominated": False},
    )

    assert acceptance["mean_abs_error_ratio_vs_fp8_jax"] == 1.05
    assert acceptance["max_abs_error_ratio_vs_fp8_jax"] == 1.05
    assert acceptance["meets_accuracy_goal"] is True
    assert acceptance["meets_speed_goal"] is True
    assert acceptance["meets_acceptance"] is True


def test_resolve_existing_artifact_paths_accepts_matching_cached_artifact(tmp_path: Path) -> None:
    bundle = ArtifactBundle(bundle_dir=tmp_path, config_name="cfg")
    onnx_path = tmp_path / "onnx" / "model_fp8.onnx"
    engine_path = tmp_path / "engine" / "model_fp8.engine"
    onnx_path.parent.mkdir(parents=True)
    engine_path.parent.mkdir(parents=True)
    onnx_path.write_text("onnx")
    engine_path.write_text("engine")

    artifact = bundle.ensure_artifact("fp8", precision="fp8")
    artifact.onnx_path = str(onnx_path)
    artifact.engine_paths["model_fp8"] = str(engine_path)
    artifact.num_steps = 10
    artifact.export_options = {
        "enable_llm_nvfp4": False,
        "quantize_attention_matmul": False,
        "num_steps": 10,
        "num_calibration_samples": 32,
    }

    resolved = _resolve_existing_artifact_paths(
        bundle,
        "fp8",
        sample_count=32,
        num_steps=10,
        quantize_attention_matmul=False,
        enable_llm_nvfp4=False,
    )

    assert resolved == (onnx_path, engine_path)


def test_resolve_existing_artifact_paths_rejects_export_option_mismatch(tmp_path: Path) -> None:
    bundle = ArtifactBundle(bundle_dir=tmp_path, config_name="cfg")
    onnx_path = tmp_path / "onnx" / "model_fp8_nvfp4.onnx"
    engine_path = tmp_path / "engine" / "model_fp8_nvfp4.engine"
    onnx_path.parent.mkdir(parents=True)
    engine_path.parent.mkdir(parents=True)
    onnx_path.write_text("onnx")
    engine_path.write_text("engine")

    artifact = bundle.ensure_artifact("fp8_nvfp4", precision="fp8_nvfp4")
    artifact.onnx_path = str(onnx_path)
    artifact.engine_paths["model_fp8_nvfp4"] = str(engine_path)
    artifact.num_steps = 10
    artifact.calibration_num_samples = 32
    artifact.export_options = {
        "enable_llm_nvfp4": True,
        "quantize_attention_matmul": False,
        "num_steps": 10,
        "num_calibration_samples": 32,
    }

    resolved = _resolve_existing_artifact_paths(
        bundle,
        "fp8_nvfp4",
        sample_count=32,
        num_steps=10,
        quantize_attention_matmul=True,
        enable_llm_nvfp4=True,
    )

    assert resolved is None


def test_resolve_existing_candidate_bundle_state_reuses_export_without_engine(tmp_path: Path) -> None:
    bundle = ArtifactBundle(bundle_dir=tmp_path, config_name="cfg")
    onnx_path = tmp_path / "onnx" / "model_fp8_nvfp4.onnx"
    onnx_path.parent.mkdir(parents=True)
    onnx_path.write_text("onnx")

    artifact = bundle.ensure_artifact("fp8_nvfp4", precision="fp8_nvfp4")
    artifact.onnx_path = str(onnx_path)
    bundle.save()

    resolved = _resolve_existing_candidate_bundle_state(tmp_path, "fp8_nvfp4")

    assert resolved is not None
    loaded_bundle, resolved_onnx, resolved_engine = resolved
    assert loaded_bundle.bundle_dir == tmp_path
    assert resolved_onnx == onnx_path
    assert resolved_engine is None
