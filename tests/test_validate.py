from types import SimpleNamespace

import numpy as np

from openpi_thor._schema import ArtifactBundle
from openpi_thor.validate import compare_tensorrt_engines
from openpi_thor.validate import _default_thresholds
from openpi_thor.validate import _artifact_precision_from_path
from openpi_thor.validate import _validation_key


def test_validation_helpers_derive_precision_and_key() -> None:
    assert _artifact_precision_from_path("/tmp/model_fp16.engine", fallback="fp8") == "fp16"
    assert _artifact_precision_from_path("/tmp/model_fp8_nvfp4.engine", fallback="fp8") == "fp8_nvfp4"
    assert _artifact_precision_from_path("/tmp/model.engine", fallback="fp8") == "fp8"
    assert _validation_key("pytorch", "/tmp/model.safetensors") == "pytorch"
    assert _validation_key("tensorrt", "/tmp/model_fp16.engine") == "tensorrt:model_fp16"
    assert _validation_key(
        "tensorrt",
        "/tmp/model_fp8.engine",
        reference_backend="tensorrt",
        reference_path="/tmp/model_fp16.engine",
    ) == "tensorrt:model_fp16:vs:model_fp8"


def test_default_thresholds_follow_selected_candidate_path(tmp_path) -> None:
    bundle = ArtifactBundle(bundle_dir=tmp_path, config_name="pi05_xlerobot_pinc_finetune", precision="fp8")
    assert _default_thresholds(bundle, "tensorrt", candidate_path="/tmp/model_fp16.engine") == {
        "min_cosine": 0.995,
        "mean_abs_error": 0.03,
        "max_abs_error": 0.12,
    }
    assert _default_thresholds(
        bundle,
        "tensorrt",
        reference_backend="tensorrt",
        reference_path="/tmp/model_fp16.engine",
        candidate_path="/tmp/model_fp8.engine",
    ) == {
        "min_cosine": 0.97,
        "mean_abs_error": 0.08,
        "max_abs_error": 0.3,
    }


def test_compare_tensorrt_engines_does_not_overwrite_recommended_engine(tmp_path, monkeypatch) -> None:
    bundle = ArtifactBundle(bundle_dir=tmp_path, config_name="pi05_xlerobot_pinc_finetune", precision="fp16")
    bundle.set_recommended_engine("/tmp/model_fp16.engine", artifact_key="fp16")
    bundle.save()

    train_config = SimpleNamespace(
        name="pi05_xlerobot_pinc_finetune",
        model=SimpleNamespace(action_horizon=2, action_dim=2),
    )

    class _Policy:
        def __init__(self, output: np.ndarray) -> None:
            self._output = output

        def infer(self, example, *, noise=None):
            return {"actions": self._output}

    def _fake_load_policy_for_backend(*args, engine_path=None, **kwargs):
        resolved_engine_path = "/tmp/model_fp16.engine" if engine_path is None else str(engine_path)
        if engine_path and "fp8" in str(engine_path):
            return _Policy(np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)), resolved_engine_path, "fp8"
        return _Policy(np.zeros((2, 2), dtype=np.float32)), resolved_engine_path, "fp16"

    monkeypatch.setattr("openpi_thor.validate._resolve_train_config", lambda config: train_config)
    monkeypatch.setattr("openpi_thor.validate._resolve_bundle", lambda bundle_dir, config_name: bundle)
    monkeypatch.setattr("openpi_thor.validate.sample_dataset_examples", lambda *args, **kwargs: [{"dataset_index": 7}])
    monkeypatch.setattr("openpi_thor.validate._load_policy_for_backend", _fake_load_policy_for_backend)

    report = compare_tensorrt_engines(
        "pi05_xlerobot_pinc_finetune",
        tmp_path,
        candidate_engine_path="/tmp/model_fp8.engine",
    )

    assert report.reference_path == "/tmp/model_fp16.engine"
    assert report.reference_precision == "fp16"
    assert bundle.get_recommended_engine_path().as_posix() == "/tmp/model_fp16.engine"
    assert "tensorrt:model_fp16:vs:model_fp8" in bundle.validation_reports
