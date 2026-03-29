from __future__ import annotations

from collections.abc import Iterable
import copy
import dataclasses
import importlib.util
from pathlib import Path
from typing import Any
from typing import Protocol

import av
import jax
import numpy as np
import torch
import torchvision
import lerobot.datasets.lerobot_dataset as lerobot_dataset

import openpi.models.model as _model
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
import openpi.transforms as _transforms

from openpi_thor._schema import CalibrationError


def _resolve_train_config(config: str | _config.TrainConfig) -> _config.TrainConfig:
    if isinstance(config, _config.TrainConfig):
        return config
    return _config.get_config(config)


def _clone_tree(data: Any) -> Any:
    return jax.tree.map(lambda x: np.array(x, copy=True), data)


def _stratified_indices(length: int, num_examples: int) -> list[int]:
    if length <= 0:
        return []
    if num_examples <= 1:
        return [0]
    if num_examples >= length:
        return list(range(length))
    values = np.linspace(0, length - 1, num=num_examples)
    return sorted({int(round(value)) for value in values})


def _decode_video_frames_pyav(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
) -> torch.Tensor:
    if not timestamps:
        raise ValueError("At least one timestamp is required to decode video frames.")

    first_ts = min(timestamps)
    last_ts = max(timestamps)
    loaded_frames: list[torch.Tensor] = []
    loaded_ts: list[float] = []

    with av.open(str(video_path), "r") as container:
        stream = container.streams.video[0]
        if stream.time_base is not None:
            offset = int(first_ts / float(stream.time_base))
            container.seek(max(offset, 0), stream=stream, any_frame=False, backward=True)

        for frame in container.decode(video=0):
            if frame.pts is None:
                continue
            frame_ts = float(frame.pts * frame.time_base)
            if frame_ts + tolerance_s < first_ts:
                continue
            array = frame.to_ndarray(format="rgb24")
            loaded_frames.append(torch.from_numpy(array).permute(2, 0, 1))
            loaded_ts.append(frame_ts)
            if frame_ts >= last_ts:
                break

    if not loaded_frames:
        raise RuntimeError(f"No frames could be decoded from {video_path}")

    query_ts = torch.tensor(timestamps, dtype=torch.float32)
    loaded_ts_tensor = torch.tensor(loaded_ts, dtype=torch.float32)
    distances = torch.cdist(query_ts[:, None], loaded_ts_tensor[:, None], p=1)
    min_distances, argmin = distances.min(1)
    within_tolerance = min_distances < tolerance_s
    assert within_tolerance.all(), (
        f"One or several query timestamps unexpectedly violate the tolerance "
        f"({min_distances[~within_tolerance]} > {tolerance_s=})."
    )
    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin]).to(torch.float32) / 255.0
    return closest_frames


def _patch_lerobot_video_decoder() -> None:
    """Install a PyAV fallback when torchvision video decoding is unavailable."""

    if hasattr(torchvision.io, "VideoReader"):
        return

    import lerobot.datasets.lerobot_dataset as lerobot_dataset
    import lerobot.datasets.video_utils as video_utils

    if getattr(video_utils, "_openpi_thor_video_patch", False):
        return

    original_torchcodec = video_utils.decode_video_frames_torchcodec

    def _decode_video_frames(video_path: Path | str, timestamps: list[float], tolerance_s: float, backend: str | None = None):
        if backend == "torchcodec" and importlib.util.find_spec("torchcodec"):
            return original_torchcodec(video_path, timestamps, tolerance_s)
        return _decode_video_frames_pyav(video_path, timestamps, tolerance_s)

    video_utils.decode_video_frames = _decode_video_frames
    lerobot_dataset.decode_video_frames = _decode_video_frames
    video_utils._openpi_thor_video_patch = True


def sample_dataset_examples(
    config: str | _config.TrainConfig,
    *,
    num_examples: int = 8,
    dataset_repo_id: str | None = None,
    dataset_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Sample and repack real dataset examples for validation or calibration."""

    train_config = _resolve_train_config(config)
    _patch_lerobot_video_decoder()
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if dataset_repo_id is not None:
        data_config = dataclasses.replace(data_config, repo_id=dataset_repo_id)
    dataset = _create_dataset_for_sampling(
        data_config,
        action_horizon=train_config.model.action_horizon,
        model_config=train_config.model,
        dataset_root=dataset_root,
    )
    repack = _transforms.compose(data_config.repack_transforms.inputs)

    examples: list[dict[str, Any]] = []
    for index in _stratified_indices(len(dataset), num_examples):
        raw = _clone_tree(dataset[index])
        repacked = repack(raw)
        repacked = {k: v for k, v in repacked.items() if k != "actions"}
        repacked["dataset_index"] = np.asarray(index)
        examples.append(repacked)
    return examples


def _create_dataset_for_sampling(
    data_config: _config.DataConfig,
    *,
    action_horizon: int,
    model_config,
    dataset_root: str | Path | None = None,
):
    if dataset_root is None:
        return _data_loader.create_torch_dataset(
            data_config,
            action_horizon=action_horizon,
            model_config=model_config,
        )

    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return _data_loader.FakeDataset(model_config, num_samples=1024)

    resolved_root = Path(dataset_root).expanduser().resolve()
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=resolved_root)
    dataset = lerobot_dataset.LeRobotDataset(
        repo_id,
        root=resolved_root,
        delta_timestamps={key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys},
    )
    if data_config.prompt_from_task:
        dataset = _data_loader.TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])
    return dataset


def _infer_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _noise_tensor(
    action_horizon: int,
    action_dim: int,
    *,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device=str(device) if device.type == "cuda" else "cpu")
    generator.manual_seed(seed)
    return torch.randn(
        1,
        action_horizon,
        action_dim,
        generator=generator,
        device=device,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )


def _prepare_policy_example(
    policy,
    example: dict[str, Any],
    *,
    device: torch.device,
    noise_seed: int,
) -> tuple[_model.Observation, torch.Tensor]:
    transformed = policy._input_transform(_clone_tree(example))  # noqa: SLF001
    transformed = {k: v for k, v in transformed.items() if k != "actions"}
    torch_inputs = jax.tree.map(lambda x: torch.from_numpy(np.asarray(x)).to(device)[None, ...], transformed)
    observation = _model.Observation.from_dict(torch_inputs)
    noise = _noise_tensor(
        policy._model.config.action_horizon,  # noqa: SLF001
        policy._model.config.action_dim,  # noqa: SLF001
        seed=noise_seed,
        device=device,
    )
    return observation, noise


@dataclasses.dataclass
class CalibrationBatches:
    """Materialized calibration batches in the shape expected by ModelOpt."""

    batches: list[tuple[_model.Observation, torch.Tensor]]

    @property
    def dataset(self) -> "CalibrationBatches":
        return self

    def __iter__(self):
        yield from self.batches

    def __len__(self) -> int:
        return len(self.batches)


class CalibrationSource(Protocol):
    """Protocol for objects that can materialize calibration batches."""

    name: str

    def materialize(
        self,
        policy,
        train_config: _config.TrainConfig,
        *,
        device: str | torch.device | None = None,
    ) -> CalibrationBatches:
        ...


@dataclasses.dataclass(frozen=True)
class IterableCalibrationSource:
    """Turn pre-built examples into calibration batches for a specific policy."""

    examples: Iterable[dict[str, Any]]
    name: str = "iterable"

    def materialize(
        self,
        policy,
        train_config: _config.TrainConfig,
        *,
        device: str | torch.device | None = None,
    ) -> CalibrationBatches:
        resolved_device = _infer_device(device)
        batches = [
            _prepare_policy_example(policy, copy.deepcopy(example), device=resolved_device, noise_seed=index)
            for index, example in enumerate(self.examples)
        ]
        if not batches:
            raise CalibrationError("Calibration source yielded zero usable examples.")
        return CalibrationBatches(batches)


@dataclasses.dataclass(frozen=True)
class LeRobotPi05CalibrationSource:
    """Sample real LeRobot examples using the host OpenPI training config."""

    config: str | _config.TrainConfig
    num_samples: int = 32
    dataset_repo_id: str | None = None
    dataset_root: str | Path | None = None
    name: str = "lerobot_pi05"

    def materialize(
        self,
        policy,
        train_config: _config.TrainConfig,
        *,
        device: str | torch.device | None = None,
    ) -> CalibrationBatches:
        examples = sample_dataset_examples(
            self.config,
            num_examples=self.num_samples,
            dataset_repo_id=self.dataset_repo_id,
            dataset_root=self.dataset_root,
        )
        return IterableCalibrationSource(examples, name=self.name).materialize(
            policy,
            train_config,
            device=device,
        )


@dataclasses.dataclass(frozen=True)
class DummyCalibrationSource:
    """Generate synthetic calibration inputs for debugging only."""

    num_samples: int = 32
    seed: int = 0
    name: str = "dummy"

    def materialize(
        self,
        policy,
        train_config: _config.TrainConfig,
        *,
        device: str | torch.device | None = None,
    ) -> CalibrationBatches:
        resolved_device = _infer_device(device)
        observation_spec, _ = train_config.model.inputs_spec(batch_size=1)
        observation_dict = jax.tree.map(
            lambda spec: np.ones(spec.shape, dtype=np.dtype(spec.dtype)),
            observation_spec.to_dict(),
        )
        batches = []
        for index in range(self.num_samples):
            torch_inputs = jax.tree.map(lambda x: torch.from_numpy(np.asarray(x)).to(resolved_device), observation_dict)
            batches.append(
                (
                    _model.Observation.from_dict(torch_inputs),
                    _noise_tensor(
                        train_config.model.action_horizon,
                        train_config.model.action_dim,
                        seed=self.seed + index,
                        device=resolved_device,
                    ),
                )
            )
        return CalibrationBatches(batches)


def build_calibration_batches(
    calibration_source: CalibrationSource,
    policy,
    train_config: _config.TrainConfig,
    *,
    device: str | torch.device | None = None,
) -> CalibrationBatches:
    """Materialize and validate calibration batches from a chosen source."""

    batches = calibration_source.materialize(policy, train_config, device=device)
    if len(batches) == 0:
        raise CalibrationError("Calibration source produced no batches.")
    return batches
