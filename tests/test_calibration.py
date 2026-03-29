import pytest
import torch

from openpi.training import config as _config

from openpi_thor._schema import CalibrationError
from openpi_thor.calibration import DummyCalibrationSource
from openpi_thor.calibration import IterableCalibrationSource
from openpi_thor.calibration import LeRobotPi05CalibrationSource


def test_empty_iterable_calibration_source_fails_closed() -> None:
    source = IterableCalibrationSource([])
    with pytest.raises(CalibrationError):
        source.materialize(policy=object(), train_config=object())


def test_dummy_calibration_source_produces_requested_batches() -> None:
    train_config = _config.get_config("debug_pi05")
    source = DummyCalibrationSource(num_samples=2, seed=7)

    batches = source.materialize(policy=None, train_config=train_config, device="cpu")

    assert len(batches) == 2
    observation, noise = batches.batches[0]
    assert observation.state.shape == (1, train_config.model.action_dim)
    assert noise.shape == (1, train_config.model.action_horizon, train_config.model.action_dim)
    assert noise.dtype in (torch.float16, torch.float32)


def test_lerobot_calibration_source_forwards_dataset_overrides(monkeypatch) -> None:
    captured = {}

    def _fake_sample_dataset_examples(config, *, num_examples, dataset_repo_id=None, dataset_root=None):
        captured["config"] = config
        captured["num_examples"] = num_examples
        captured["dataset_repo_id"] = dataset_repo_id
        captured["dataset_root"] = dataset_root
        return [{"state": torch.zeros(14).numpy(), "prompt": "test"}]

    monkeypatch.setattr("openpi_thor.calibration.sample_dataset_examples", _fake_sample_dataset_examples)
    monkeypatch.setattr(
        "openpi_thor.calibration.IterableCalibrationSource.materialize",
        lambda self, policy, train_config, device=None: "ok",
    )

    source = LeRobotPi05CalibrationSource(
        config="pi05_xlerobot_pinc_finetune",
        num_samples=5,
        dataset_repo_id="custom/repo",
        dataset_root="/tmp/dataset-root",
    )

    assert source.materialize(policy=object(), train_config=object(), device="cpu") == "ok"
    assert captured["num_examples"] == 5
    assert captured["dataset_repo_id"] == "custom/repo"
    assert captured["dataset_root"] == "/tmp/dataset-root"
