from openpi_thor._schema import ArtifactBundle
from openpi_thor.validate import _default_thresholds
from openpi_thor.validate import _artifact_precision_from_path
from openpi_thor.validate import _validation_key


def test_validation_helpers_derive_precision_and_key() -> None:
    assert _artifact_precision_from_path("/tmp/model_fp16.engine", fallback="fp8") == "fp16"
    assert _artifact_precision_from_path("/tmp/model_fp8_nvfp4.engine", fallback="fp8") == "fp8_nvfp4"
    assert _artifact_precision_from_path("/tmp/model.engine", fallback="fp8") == "fp8"
    assert _validation_key("pytorch", "/tmp/model.safetensors") == "pytorch"
    assert _validation_key("tensorrt", "/tmp/model_fp16.engine") == "tensorrt:model_fp16"


def test_default_thresholds_follow_selected_candidate_path(tmp_path) -> None:
    bundle = ArtifactBundle(bundle_dir=tmp_path, config_name="pi05_xlerobot_pinc_finetune", precision="fp8")
    assert _default_thresholds(bundle, "tensorrt", candidate_path="/tmp/model_fp16.engine") == {
        "min_cosine": 0.995,
        "mean_abs_error": 0.03,
        "max_abs_error": 0.12,
    }
