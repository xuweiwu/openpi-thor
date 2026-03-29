import inspect
import types

import pytest
import torch

from openpi_thor._schema import ArtifactBundle
from openpi_thor._schema import ValidationReport
from openpi_thor.runtime import _apply_state_dict_with_report
from openpi_thor.runtime import _candidate_engine_path
from openpi_thor.runtime import _ensure_ready_for_tensorrt
from openpi_thor.runtime import _install_tensorrt_sample_actions
import openpi_thor.runtime as _runtime


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)
        self.action_out_proj = torch.nn.Linear(1, 1)
        self.config = types.SimpleNamespace(action_horizon=4, action_dim=3)

    def sample_actions(self, device, observation, noise=None, num_steps=10):  # noqa: ARG002
        return "pytorch"


def test_apply_state_dict_with_report_fails_closed_on_mismatch() -> None:
    model = TinyModel()
    state_dict = {
        "linear.weight": torch.zeros_like(model.linear.weight),
        "linear.bias": torch.zeros(3),
        "unexpected": torch.ones(1),
    }

    report = _apply_state_dict_with_report(model, state_dict)

    assert not report.clean
    assert report.shape_mismatches[0].key == "linear.bias"
    assert "unexpected" in report.unexpected_keys
    assert "linear.bias" in report.missing_keys


def test_install_tensorrt_sample_actions_refreshes_cached_policy_reference() -> None:
    model = TinyModel()
    policy = types.SimpleNamespace(_model=model, _sample_actions=model.sample_actions)
    engine = types.SimpleNamespace(in_meta=[])

    _install_tensorrt_sample_actions(policy, engine)

    assert policy._sample_actions is model.sample_actions
    assert model.trt_engine is engine
    assert not hasattr(model, "action_out_proj")


def test_unvalidated_fp8_bundle_is_rejected_for_serving(tmp_path) -> None:
    bundle = ArtifactBundle(bundle_dir=tmp_path, config_name="pi05_xlerobot_pinc_finetune", precision="fp8")
    with pytest.raises(Exception):
        _ensure_ready_for_tensorrt(bundle, require_validated=True)

    bundle.set_validation_report(
        "tensorrt",
        ValidationReport(
            reference_backend="jax",
            candidate_backend="tensorrt",
            config_name=bundle.config_name,
            passed=True,
        ),
    )
    _ensure_ready_for_tensorrt(bundle, require_validated=True)


def test_candidate_engine_path_prefers_recommended_engine(tmp_path) -> None:
    bundle = ArtifactBundle(bundle_dir=tmp_path, config_name="pi05_xlerobot_pinc_finetune")
    engine_a = tmp_path / "engine_a.engine"
    engine_b = tmp_path / "engine_b.engine"
    bundle.set_engine_path("a", engine_a, artifact_key="fp16")
    bundle.set_engine_path("b", engine_b, artifact_key="fp16")
    bundle.set_recommended_engine(engine_b, artifact_key="fp16")

    assert _candidate_engine_path(bundle, None) == engine_b


def test_runtime_no_longer_depends_on_openpi_on_thor() -> None:
    assert "openpi_on_thor" not in inspect.getsource(_runtime)
