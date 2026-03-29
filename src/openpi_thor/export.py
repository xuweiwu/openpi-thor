from __future__ import annotations

from collections.abc import Iterable
import dataclasses
import os
from pathlib import Path
import re
import subprocess
import types

import torch

from openpi.models import gemma as _gemma
from openpi.models import model as _model
from openpi.models_pytorch import pi0_pytorch as _pi0_pytorch
from openpi.training import config as _config

from openpi_thor._schema import ArtifactBundle
from openpi_thor._schema import CalibrationError
from openpi_thor._schema import ExportOptions
from openpi_thor.calibration import DummyCalibrationSource
from openpi_thor.calibration import LeRobotPi05CalibrationSource
from openpi_thor.calibration import build_calibration_batches
from openpi_thor.calibration import CalibrationSource
from openpi_thor.compat import prepare_runtime
from openpi_thor.runtime import load_pytorch_bundle


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


def _system_blackwell_ptxas_candidates() -> list[Path]:
    return [Path("/usr/local/cuda/bin/ptxas")]


def _bundled_blackwell_ptxas_path() -> Path | None:
    try:
        import triton
    except ImportError:
        return None
    return Path(triton.__file__).resolve().parent / "backends" / "nvidia" / "bin" / "ptxas-blackwell"


def _read_ptxas_release(path: Path) -> tuple[int, int] | None:
    try:
        output = subprocess.check_output([str(path), "--version"], stderr=subprocess.STDOUT, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    match = re.search(r"release (\d+)\.(\d+)", output)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _prefer_system_blackwell_ptxas() -> str | None:
    env_key = "TRITON_PTXAS_BLACKWELL_PATH"
    if os.environ.get(env_key):
        return os.environ[env_key]

    bundled_path = _bundled_blackwell_ptxas_path()
    bundled_version = _read_ptxas_release(bundled_path) if bundled_path is not None else None
    for candidate in _system_blackwell_ptxas_candidates():
        if not candidate.exists():
            continue
        candidate_version = _read_ptxas_release(candidate)
        if candidate_version is None:
            continue
        if bundled_version is None or candidate_version > bundled_version:
            os.environ[env_key] = str(candidate)
            return str(candidate)
    return None


class QuantizedMatMul(torch.nn.Module):
    """MatMul wrapper that lets ModelOpt attach quantizers to attention ops."""

    def __init__(self):
        super().__init__()
        self.input1_quantizer = None
        self.input2_quantizer = None
        self._quantizers_created = False

    def _create_quantizers(self):
        from modelopt.torch.quantization.config import QuantizerAttributeConfig
        from modelopt.torch.quantization.nn import TensorQuantizer

        if not self._quantizers_created:
            self.input1_quantizer = TensorQuantizer(QuantizerAttributeConfig(num_bits=(4, 3)))
            self.input2_quantizer = TensorQuantizer(QuantizerAttributeConfig(num_bits=(4, 3)))
            self.input1_quantizer.enable_calib()
            self.input1_quantizer.disable_quant()
            self.input2_quantizer.enable_calib()
            self.input2_quantizer.disable_quant()
            self._quantizers_created = True

    def forward(self, input1, input2):
        if not self._quantizers_created:
            self._create_quantizers()
        input1 = self.input1_quantizer(input1)
        input2 = self.input2_quantizer(input2)
        return torch.matmul(input1, input2)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def quantized_eager_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,  # noqa: ARG001
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    if not hasattr(module, "qk_matmul"):
        module.add_module("qk_matmul", QuantizedMatMul())
    if not hasattr(module, "av_matmul"):
        module.add_module("av_matmul", QuantizedMatMul())
    attn_weights = module.qk_matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
    attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = torch.dropout(attn_weights, p=dropout, train=module.training)
    attn_output = module.av_matmul(attn_weights, value_states)
    return attn_output.transpose(1, 2).contiguous(), attn_weights


def replace_attention_with_quantized_version() -> None:
    from transformers.models.gemma import modeling_gemma

    if not hasattr(modeling_gemma, "_openpi_thor_original_eager_attention_forward"):
        modeling_gemma._openpi_thor_original_eager_attention_forward = modeling_gemma.eager_attention_forward
    modeling_gemma.eager_attention_forward = quantized_eager_attention_forward


def _create_observation_from_inputs(images, img_masks, state, lang_tokens, lang_masks):
    images_dict = {key: images[:, i * 3 : (i + 1) * 3] for i, key in enumerate(_model.IMAGE_KEYS)}
    image_masks_dict = {key: img_masks[:, i] for i, key in enumerate(_model.IMAGE_KEYS)}
    return _model.Observation(
        images=images_dict,
        image_masks=image_masks_dict,
        state=state,
        tokenized_prompt=lang_tokens,
        tokenized_prompt_mask=lang_masks,
    )


def postprocess_onnx_model(onnx_path: str | Path, *, enable_llm_nvfp4: bool = False) -> None:
    """Normalize exported ONNX artifacts and optionally rewrite NVFP4 QDQ nodes."""

    import onnx
    import onnx_graphsurgeon as gs
    from onnx.external_data_helper import convert_model_to_external_data

    onnx_path = Path(onnx_path)
    onnx_dir = onnx_path.parent
    keep_names = {onnx_path.name, onnx_path.with_suffix(".data").name}
    onnx_model = onnx.load(str(onnx_path), load_external_data=True)

    if enable_llm_nvfp4:
        from modelopt.onnx.quantization.qdq_utils import fp4qdq_to_2dq

        onnx_model = fp4qdq_to_2dq(onnx_model, verbose=True)
        graph = gs.import_onnx(onnx_model)
        graph.cleanup().toposort()
        onnx_model = gs.export_onnx(graph)

    convert_model_to_external_data(
        onnx_model,
        all_tensors_to_one_file=True,
        location=onnx_path.with_suffix(".data").name,
    )
    onnx.save(onnx_model, str(onnx_path))

    # Torch ONNX export may leave many stale external-data shard files behind
    # after consolidation. Keep named model artifacts and remove shard junk.
    for path in onnx_dir.iterdir():
        if not path.is_file():
            continue
        if path.name in keep_names:
            continue
        if path.suffix in {".onnx", ".data", ".engine", ".json"}:
            continue
        path.unlink()


class ONNXWrapper(torch.nn.Module):
    """Wrap the policy sampler in a flat tensor-only interface for ONNX export."""

    def __init__(self, model: torch.nn.Module, num_steps: int):
        super().__init__()
        self.model = model
        self.num_steps = num_steps

    def forward(self, images, img_masks, lang_tokens, lang_masks, state, noise):
        observation = _create_observation_from_inputs(images, img_masks, state, lang_tokens, lang_masks)
        return self.model.sample_actions(images.device, observation, noise=noise, num_steps=self.num_steps)


def _create_dummy_inputs(
    model_device: torch.device,
    model_config,
    compute_dtype: torch.dtype = torch.float16,
):
    num_images = len(_model.IMAGE_KEYS)
    image_size = _model.IMAGE_RESOLUTION[0]
    return (
        torch.randn(1, num_images * 3, image_size, image_size, dtype=compute_dtype, device=model_device),
        torch.ones(1, num_images, dtype=torch.bool, device=model_device),
        torch.randint(0, _gemma.PALIGEMMA_VOCAB_SIZE, (1, model_config.max_token_len), dtype=torch.long, device=model_device),
        torch.ones(1, model_config.max_token_len, dtype=torch.bool, device=model_device),
        torch.randn(1, model_config.action_dim, dtype=torch.float32, device=model_device),
        torch.randn(1, model_config.action_horizon, model_config.action_dim, dtype=torch.float32, device=model_device),
    )


def _attention_mask_fill_value(dtype: torch.dtype) -> float:
    if dtype == torch.float16:
        # TensorRT clips the training-time sentinel (~-2.38e38) down to the
        # half range anyway. Emit the half-safe value directly during export so
        # the graph stays numerically consistent across backends.
        return float(torch.finfo(torch.float16).min)
    return -2.3819763e38


_FLOAT32_EXPORT_PARAM_SELECTORS = (
    "vision_tower.vision_model.embeddings.patch_embedding.weight",
    "vision_tower.vision_model.embeddings.patch_embedding.bias",
    "vision_tower.vision_model.embeddings.position_embedding.weight",
    "input_layernorm",
    "post_attention_layernorm",
    "model.norm",
    "action_in_proj",
    "action_out_proj",
    "time_mlp_in",
    "time_mlp_out",
    "action_time_mlp_in",
    "action_time_mlp_out",
)

_FLOAT32_EXPORT_BUFFER_SELECTORS = (
    "rotary_emb.inv_freq",
    "rotary_emb.original_inv_freq",
)


def _module_and_attr(root: torch.nn.Module, dotted_name: str) -> tuple[torch.nn.Module, str]:
    parts = dotted_name.split(".")
    module = root
    for part in parts[:-1]:
        module = getattr(module, part)
    return module, parts[-1]


def _keep_export_float32(name: str, selectors: tuple[str, ...]) -> bool:
    return any(selector in name for selector in selectors)


def prepare_model_for_export_precision(
    model: torch.nn.Module,
    *,
    compute_dtype: torch.dtype = torch.float16,
) -> torch.nn.Module:
    """Preserve the model's float32 stability islands while lowering the rest.

    The normal PyTorch inference path keeps selected weights and buffers in
    float32 for numerical stability. Flattening the whole model to float16
    destroys that layout and appears to accumulate large KV-cache drift on Thor.
    """

    for name, param in model.named_parameters():
        if not param.is_floating_point():
            continue
        target_dtype = (
            torch.float32 if _keep_export_float32(name, _FLOAT32_EXPORT_PARAM_SELECTORS) else compute_dtype
        )
        if param.dtype != target_dtype:
            param.data = param.data.to(dtype=target_dtype)

    for name, buf in model.named_buffers():
        if not torch.is_floating_point(buf):
            continue
        target_dtype = (
            torch.float32 if _keep_export_float32(name, _FLOAT32_EXPORT_BUFFER_SELECTORS) else compute_dtype
        )
        if buf.dtype != target_dtype:
            module, attr = _module_and_attr(model, name)
            module.register_buffer(attr, buf.to(dtype=target_dtype), persistent=name not in module._non_persistent_buffers_set)

    return model


def patch_model_for_export(model: torch.nn.Module, *, compute_dtype: torch.dtype = torch.float16) -> torch.nn.Module:
    """Patch the PyTorch sampler into an ONNX-export-friendly inference loop."""

    model.compute_dtype = compute_dtype

    vision_embeddings = model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings

    def prepare_attention_masks_4d_hook(self, att_2d_masks):
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        # Keep the mask arithmetic in float32 like the stable PyTorch path, but
        # clamp the sentinel to the fp16-safe range so TensorRT does not later
        # clip a huge constant differently across backends.
        target_dtype = torch.float32
        fill_value = _attention_mask_fill_value(getattr(self, "compute_dtype", torch.float16))
        zeros = torch.zeros((), dtype=target_dtype, device=att_2d_masks.device)
        neg = torch.tensor(fill_value, dtype=target_dtype, device=att_2d_masks.device)
        return torch.where(att_2d_masks_4d, zeros, neg)

    def make_att_2d_masks_hook(pad_masks, att_masks):
        att_masks_int64 = att_masks.to(dtype=torch.int64)
        cumsum = torch.cumsum(att_masks_int64, dim=1)
        att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
        pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
        return att_2d_masks & pad_2d_masks

    def sample_noise_hook(self, shape, device):
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)

    original_vision_embeddings_forward = vision_embeddings.forward

    def vision_embeddings_forward_hook(self, pixel_values, interpolate_pos_encoding=False):
        embeddings = original_vision_embeddings_forward(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        target_dtype = model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers[
            0
        ].self_attn.q_proj.weight.dtype
        if target_dtype != torch.float32 and embeddings.dtype != target_dtype:
            embeddings = embeddings.to(dtype=target_dtype)
        return embeddings

    def sample_time_hook(self, bsize, device):
        time_beta = _pi0_pytorch.sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def sample_actions_hook(self, device, observation, noise=None, num_steps=10):
        bsize = observation.state.shape[0]
        if noise is None:
            shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(shape, device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_q_dtype = self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
        if prefix_embs.dtype != prefix_q_dtype:
            prefix_embs = prefix_embs.to(dtype=prefix_q_dtype)
        prefix_att_2d_masks = make_att_2d_masks_hook(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks.to(dtype=torch.int64), dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(state, prefix_pad_masks, past_key_values, x_t, expanded_time)
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def denoise_step_hook(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)
        suffix_q_dtype = self.paligemma_with_expert.gemma_expert.model.layers[0].self_attn.q_proj.weight.dtype
        adarms_dtype = self.paligemma_with_expert.gemma_expert.model.layers[0].input_layernorm.dense.weight.dtype
        if suffix_embs.dtype != suffix_q_dtype:
            suffix_embs = suffix_embs.to(dtype=suffix_q_dtype)
        if adarms_cond is not None and adarms_cond.dtype != adarms_dtype:
            adarms_cond = adarms_cond.to(dtype=adarms_dtype)
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks_hook(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks.to(dtype=torch.int64), dim=1) - 1
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        suffix_out = outputs_embeds[1][:, -self.config.action_horizon :].to(dtype=self.action_out_proj.weight.dtype)
        return self.action_out_proj(suffix_out)

    model.sample_noise = types.MethodType(sample_noise_hook, model)
    model.sample_time = types.MethodType(sample_time_hook, model)
    model.sample_actions = types.MethodType(sample_actions_hook, model)
    model.denoise_step = types.MethodType(denoise_step_hook, model)
    model._prepare_attention_masks_4d = types.MethodType(prepare_attention_masks_4d_hook, model)
    vision_embeddings.forward = types.MethodType(vision_embeddings_forward_hook, vision_embeddings)
    return model


def quantize_model(
    model: torch.nn.Module,
    *,
    calibration_data: Iterable[tuple[_model.Observation, torch.Tensor]],
    num_steps: int,
    enable_llm_nvfp4: bool,
    quantize_attention_matmul: bool,
) -> torch.nn.Module:
    """Apply ModelOpt quantization to a patched PyTorch policy model."""

    if enable_llm_nvfp4:
        _prefer_system_blackwell_ptxas()
    import modelopt.torch.quantization as mtq

    if quantize_attention_matmul:
        replace_attention_with_quantized_version()

    quant_cfg = mtq.FP8_DEFAULT_CFG
    quant_cfg["quant_cfg"]["nn.Conv2d"] = {"*": {"enable": False}}

    if enable_llm_nvfp4:
        quant_cfg["quant_cfg"]["paligemma_with_expert.paligemma.model.language_model.layers.*"] = {
            "num_bits": (2, 1),
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
            "axis": None,
            "enable": True,
        }
        quant_cfg["quant_cfg"][
            "paligemma_with_expert.paligemma.model.language_model.layers.*.output_quantizer"
        ] = {
            "num_bits": (2, 1),
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
            "axis": None,
            "enable": False,
        }

    def forward_loop(mdl):
        mdl.eval()
        device = next(mdl.parameters()).device
        for observation, noise in calibration_data:
            with torch.no_grad():
                mdl.sample_actions(device, observation, noise=noise, num_steps=num_steps)

    quantized_model = mtq.quantize(model, quant_cfg, forward_loop=forward_loop)
    if enable_llm_nvfp4:
        from modelopt.torch.quantization.utils import is_quantized_linear

        for module in quantized_model.modules():
            if isinstance(module, torch.nn.Linear):
                assert is_quantized_linear(module)
                module.input_quantizer._trt_high_precision_dtype = "Half"  # noqa: SLF001
                module.input_quantizer._onnx_quantizer_type = "dynamic"  # noqa: SLF001
                module.output_quantizer._onnx_quantizer_type = "dynamic"  # noqa: SLF001
                module.weight_quantizer._onnx_quantizer_type = "static"  # noqa: SLF001
    return quantized_model


def _artifact_key(options: ExportOptions) -> str:
    if options.precision.lower() == "fp8" and options.enable_llm_nvfp4:
        return "fp8_nvfp4"
    return options.precision.lower()


def export_to_onnx_bundle(
    config: str | _config.TrainConfig,
    bundle_dir: str | Path,
    *,
    options: ExportOptions,
    calibration_source: CalibrationSource | None = None,
    default_prompt: str | None = None,
    pytorch_device: str | None = None,
    dataset_repo_id: str | None = None,
    dataset_root: str | Path | None = None,
) -> ArtifactBundle:
    """Export a bundle to ONNX and record the resulting artifact metadata."""

    import onnx

    prepare_runtime()
    train_config = _resolve_train_config(config)
    bundle = _resolve_bundle(bundle_dir, config_name=train_config.name)
    policy, _ = load_pytorch_bundle(
        train_config,
        bundle.bundle_dir,
        default_prompt=default_prompt,
        pytorch_device=pytorch_device,
    )
    model = policy._model  # noqa: SLF001
    model.eval()
    model = patch_model_for_export(model, compute_dtype=torch.float16)
    model = prepare_model_for_export_precision(model, compute_dtype=torch.float16)

    device = next(model.parameters()).device
    if options.precision.lower() == "fp8":
        if calibration_source is None:
            if options.allow_dummy_calibration:
                calibration_source = DummyCalibrationSource(num_samples=options.num_calibration_samples)
            else:
                calibration_source = LeRobotPi05CalibrationSource(
                    config=config,
                    num_samples=options.num_calibration_samples,
                    dataset_repo_id=dataset_repo_id,
                    dataset_root=dataset_root,
                )
        if isinstance(calibration_source, DummyCalibrationSource) and not options.allow_dummy_calibration:
            raise CalibrationError("Dummy calibration is opt-in only for FP8/NVFP4 export.")
        calibration_batches = build_calibration_batches(
            calibration_source,
            policy,
            train_config,
            device=device,
        )
        if len(calibration_batches) == 0:
            raise CalibrationError("Calibration source produced zero valid samples.")
        model = quantize_model(
            model,
            calibration_data=calibration_batches,
            num_steps=options.num_steps,
            enable_llm_nvfp4=options.enable_llm_nvfp4,
            quantize_attention_matmul=options.quantize_attention_matmul,
        )
        calibration_source_name = calibration_source.name
        calibration_num_samples = len(calibration_batches)
        bundle.calibration_source = calibration_source_name
        bundle.calibration_num_samples = calibration_num_samples
    elif calibration_source is not None:
        calibration_source_name = calibration_source.name
        calibration_num_samples = None
        bundle.calibration_source = calibration_source_name
    else:
        calibration_source_name = None
        calibration_num_samples = None
        bundle.calibration_source = None
        bundle.calibration_num_samples = None

    dummy_inputs = _create_dummy_inputs(device, train_config.model, torch.float16)
    wrapped_model = ONNXWrapper(model, options.num_steps)
    onnx_dir = bundle.bundle_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    key = _artifact_key(options)
    onnx_path = onnx_dir / f"model_{key}.onnx"
    with torch.no_grad():
        torch.onnx.export(
            wrapped_model,
            dummy_inputs,
            str(onnx_path),
            opset_version=19,
            dynamo=False,
            do_constant_folding=True,
            input_names=["images", "img_masks", "lang_tokens", "lang_masks", "state", "noise"],
            output_names=["actions"],
            dynamic_axes={
                "images": {0: "batch_size"},
                "img_masks": {0: "batch_size"},
                "lang_tokens": {0: "batch_size", 1: "seq_len"},
                "lang_masks": {0: "batch_size", 1: "seq_len"},
                "state": {0: "batch_size"},
                "noise": {0: "batch_size"},
                "actions": {0: "batch_size"},
            },
        )
    postprocess_onnx_model(onnx_path, enable_llm_nvfp4=options.enable_llm_nvfp4)
    # Large external-data models must be checked by path, not by loading the
    # whole protobuf into memory first.
    onnx.checker.check_model(str(onnx_path))

    bundle.precision = key
    bundle.num_steps = options.num_steps
    bundle.set_onnx_path(
        key,
        onnx_path,
        precision=key,
        num_steps=options.num_steps,
        calibration_source=calibration_source_name,
        calibration_num_samples=calibration_num_samples,
        export_options=dataclasses.asdict(options),
    )
    bundle.write_report(
        f"export_{key}",
        {
            "phase": "export_to_onnx",
            "artifact_key": key,
            "config_name": train_config.name,
            "bundle_dir": str(bundle.bundle_dir),
            "onnx_path": str(onnx_path),
            "precision": key,
            "num_steps": options.num_steps,
            "enable_llm_nvfp4": options.enable_llm_nvfp4,
            "quantize_attention_matmul": options.quantize_attention_matmul,
            "calibration_source": calibration_source_name,
            "calibration_num_samples": calibration_num_samples,
            "dataset_repo_id": dataset_repo_id,
            "dataset_root": str(dataset_root) if dataset_root is not None else None,
            "export_options": dataclasses.asdict(options),
        },
        artifact_key=key,
        report_key="export",
    )
    bundle.save()
    return bundle
