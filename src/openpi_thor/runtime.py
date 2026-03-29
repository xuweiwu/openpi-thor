from __future__ import annotations

from functools import partial
import dataclasses
import logging
from pathlib import Path
from typing import Any

import safetensors.torch
import torch

from openpi.models import model as _model
from openpi.models_pytorch import pi0_pytorch as _pi0_pytorch
from openpi.policies import policy as _policy
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as _transforms

from openpi_thor._schema import ArtifactBundle
from openpi_thor._schema import CheckpointLoadError
from openpi_thor._schema import CheckpointLoadReport
from openpi_thor._schema import ShapeMismatch
from openpi_thor._schema import ValidationError
from openpi_thor.compat import prepare_runtime

logger = logging.getLogger(__name__)


_KNOWN_TIED_WEIGHT_ALIASES: dict[str, tuple[str, ...]] = {
    "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight": (
        "paligemma_with_expert.paligemma.lm_head.weight",
    ),
}


def _resolve_train_config(config: str | _config.TrainConfig) -> _config.TrainConfig:
    if isinstance(config, _config.TrainConfig):
        return config
    return _config.get_config(config)


def _resolve_bundle(bundle_dir: str | Path, *, config_name: str) -> ArtifactBundle:
    bundle_path = Path(bundle_dir).expanduser().resolve()
    metadata_path = bundle_path / "openpi_thor_bundle.json"
    if metadata_path.exists():
        return ArtifactBundle.load(bundle_path)
    bundle = ArtifactBundle(bundle_dir=bundle_path, config_name=config_name)
    bundle.save()
    return bundle


def _resolve_device(device: str | None) -> str:
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _instantiate_model(train_config: _config.TrainConfig) -> torch.nn.Module:
    model_config = train_config.model
    if hasattr(model_config, "pytorch_compile_mode"):
        model_config = dataclasses.replace(model_config, pytorch_compile_mode=None)
    return _pi0_pytorch.PI0Pytorch(config=model_config)


def _apply_state_dict_with_report(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> CheckpointLoadReport:
    model_state = model.state_dict()
    compatible_state: dict[str, torch.Tensor] = {}
    unexpected_keys: list[str] = []
    shape_mismatches: list[ShapeMismatch] = []

    for key, value in state_dict.items():
        if key not in model_state:
            unexpected_keys.append(key)
            continue
        if tuple(value.shape) != tuple(model_state[key].shape):
            shape_mismatches.append(
                ShapeMismatch(
                    key=key,
                    checkpoint_shape=tuple(value.shape),
                    model_shape=tuple(model_state[key].shape),
                )
            )
            continue
        compatible_state[key] = value

    missing_keys = sorted(set(model_state) - set(compatible_state))
    for key in list(missing_keys):
        for alias in _KNOWN_TIED_WEIGHT_ALIASES.get(key, ()):
            if alias not in state_dict:
                continue
            if tuple(state_dict[alias].shape) != tuple(model_state[key].shape):
                continue
            compatible_state[key] = state_dict[alias]
            missing_keys.remove(key)
            break

    incompatible = model.load_state_dict(compatible_state, strict=False)
    missing_keys = sorted(set(missing_keys) | set(incompatible.missing_keys))
    unexpected_keys = sorted(set(unexpected_keys) | set(incompatible.unexpected_keys))

    return CheckpointLoadReport(
        total_checkpoint_keys=len(state_dict),
        loaded_keys=len(compatible_state),
        unexpected_keys=unexpected_keys,
        missing_keys=missing_keys,
        shape_mismatches=shape_mismatches,
        fail_closed=True,
        clean=not (unexpected_keys or missing_keys or shape_mismatches),
    )


def _report_to_error(report: CheckpointLoadReport, *, weight_path: Path) -> CheckpointLoadError:
    parts = [f"Unsafe checkpoint load refused for {weight_path}."]
    if report.unexpected_keys:
        parts.append(f"unexpected={len(report.unexpected_keys)}")
    if report.missing_keys:
        parts.append(f"missing={len(report.missing_keys)}")
    if report.shape_mismatches:
        parts.append(f"shape_mismatches={len(report.shape_mismatches)}")
    return CheckpointLoadError(" ".join(parts))


def _load_norm_stats(train_config: _config.TrainConfig, bundle: ArtifactBundle) -> dict | None:
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.asset_id is None:
        return None
    assets_dir = bundle.assets_dir if bundle.assets_dir.exists() else None
    if assets_dir is not None:
        return _checkpoints.load_norm_stats(assets_dir, data_config.asset_id)
    return data_config.norm_stats


def _build_policy(
    train_config: _config.TrainConfig,
    model: torch.nn.Module,
    *,
    norm_stats: dict | None,
    pytorch_device: str,
    default_prompt: str | None,
) -> _policy.Policy:
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    return _policy.Policy(
        model,
        transforms=[
            _transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            _transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ],
        sample_kwargs={},
        metadata=train_config.policy_metadata,
        pytorch_device=pytorch_device,
        is_pytorch=True,
    )


def load_pytorch_bundle(
    config: str | _config.TrainConfig,
    bundle_dir: str | Path,
    *,
    default_prompt: str | None = None,
    allow_compatibility_fallback: bool = False,
    pytorch_device: str | None = None,
) -> tuple[_policy.Policy, CheckpointLoadReport]:
    prepare_runtime()
    train_config = _resolve_train_config(config)
    bundle = _resolve_bundle(bundle_dir, config_name=train_config.name)
    weight_path = bundle.weight_path
    if not weight_path.exists():
        raise FileNotFoundError(f"Converted PyTorch weights not found at {weight_path}")

    model = _instantiate_model(train_config)
    state_dict = safetensors.torch.load_file(str(weight_path))
    report = _apply_state_dict_with_report(model, state_dict)
    if report.has_issues and not allow_compatibility_fallback:
        raise _report_to_error(report, weight_path=weight_path)

    if hasattr(model, "paligemma_with_expert"):
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")

    resolved_device = _resolve_device(pytorch_device)
    policy = _build_policy(
        train_config,
        model,
        norm_stats=_load_norm_stats(train_config, bundle),
        pytorch_device=resolved_device,
        default_prompt=default_prompt,
    )

    bundle.checkpoint_load_report = report
    bundle.save()
    return policy, report


def _candidate_engine_path(bundle: ArtifactBundle, engine_path: str | Path | None) -> Path:
    if engine_path is not None:
        return Path(engine_path).expanduser().resolve()
    if recommended := bundle.get_recommended_engine_path():
        return recommended.expanduser().resolve()
    if bundle.engine_paths:
        return Path(next(iter(bundle.engine_paths.values()))).expanduser().resolve()
    raise FileNotFoundError("No TensorRT engine path was provided and the bundle has no recorded engine.")


def _bundle_is_validated(bundle: ArtifactBundle) -> bool:
    return any(report.passed for report in bundle.validation_reports.values())


def _ensure_ready_for_tensorrt(bundle: ArtifactBundle, *, require_validated: bool) -> None:
    if not require_validated:
        return
    if bundle.precision and bundle.precision.startswith("fp8") and not _bundle_is_validated(bundle):
        raise ValidationError(
            "Refusing to serve an FP8/NVFP4 bundle without a passing validation report. "
            "Run openpi-thor validate first or pass require_validated=False."
        )


def _binding_dtypes(engine) -> dict[str, torch.dtype]:
    return {name: dtype for name, _, dtype in engine.in_meta}


def _prepare_trt_inputs(observation: _model.Observation[torch.Tensor], *, device: str, dtypes: dict[str, torch.dtype]):
    images = torch.cat([observation.images[key] for key in _model.IMAGE_KEYS], dim=1)
    img_masks = torch.stack([observation.image_masks[key] for key in _model.IMAGE_KEYS], dim=1)
    bindings = {
        "images": images.to(device=device, dtype=dtypes["images"]).contiguous(),
        "img_masks": img_masks.to(device=device, dtype=dtypes["img_masks"]).contiguous(),
        "lang_tokens": observation.tokenized_prompt.to(device=device, dtype=dtypes["lang_tokens"]).contiguous(),
        "lang_masks": observation.tokenized_prompt_mask.to(device=device, dtype=dtypes["lang_masks"]).contiguous(),
        "state": observation.state.to(device=device, dtype=dtypes["state"]).contiguous(),
    }
    return bindings


def _tensorrt_sample_actions(model, device, observation, noise=None, num_steps=None):  # noqa: ARG001
    dtypes = _binding_dtypes(model.trt_engine)
    bindings = _prepare_trt_inputs(observation, device=device, dtypes=dtypes)
    if noise is None:
        noise = torch.randn(
            observation.state.shape[0],
            model.config.action_horizon,
            model.config.action_dim,
            device=device,
            dtype=dtypes["noise"],
        )
    elif not isinstance(noise, torch.Tensor):
        noise = torch.from_numpy(noise)
    if noise.ndim == 2:
        noise = noise[None, ...]
    bindings["noise"] = noise.to(device=device, dtype=dtypes["noise"]).contiguous()

    for name, tensor in bindings.items():
        model.trt_engine.set_runtime_tensor_shape(name, tensor.shape)

    outputs = model.trt_engine(**bindings)
    return outputs["actions"]


def _free_pytorch_submodules(model: torch.nn.Module) -> None:
    removable = (
        "paligemma_with_expert",
        "time_mlp_in",
        "time_mlp_out",
        "state_proj",
        "action_time_mlp_in",
        "action_time_mlp_out",
        "action_in_proj",
        "action_out_proj",
    )
    for attr in removable:
        if hasattr(model, attr):
            delattr(model, attr)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _install_tensorrt_sample_actions(policy: _policy.Policy, engine) -> None:
    model = policy._model  # noqa: SLF001
    model.trt_engine = engine
    if not hasattr(model, "_original_sample_actions"):
        model._original_sample_actions = model.sample_actions
    trt_sample_actions = partial(_tensorrt_sample_actions, model)
    model.sample_actions = trt_sample_actions
    if hasattr(policy, "_sample_actions"):
        policy._sample_actions = trt_sample_actions  # noqa: SLF001
    _free_pytorch_submodules(model)


def load_tensorrt_policy(
    config: str | _config.TrainConfig,
    bundle_dir: str | Path,
    engine_path: str | Path | None = None,
    *,
    require_validated: bool = True,
    default_prompt: str | None = None,
    allow_compatibility_fallback: bool = False,
    pytorch_device: str | None = None,
):
    prepare_runtime()
    train_config = _resolve_train_config(config)
    bundle = _resolve_bundle(bundle_dir, config_name=train_config.name)
    _ensure_ready_for_tensorrt(bundle, require_validated=require_validated)
    resolved_engine_path = _candidate_engine_path(bundle, engine_path)
    if not resolved_engine_path.exists():
        raise FileNotFoundError(f"TensorRT engine not found at {resolved_engine_path}")

    policy, _ = load_pytorch_bundle(
        train_config,
        bundle.bundle_dir,
        default_prompt=default_prompt,
        allow_compatibility_fallback=allow_compatibility_fallback,
        pytorch_device=pytorch_device,
    )

    from openpi_thor import trt_torch

    engine = trt_torch.Engine(str(resolved_engine_path))
    logger.info("Attaching TensorRT engine %s", resolved_engine_path)
    _install_tensorrt_sample_actions(policy, engine)
    return policy
