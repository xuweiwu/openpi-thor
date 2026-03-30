"""Internal helpers for deeper fp8/NVFP4 debugging on Jetson AGX Thor."""

from __future__ import annotations

import copy
import contextlib
import dataclasses
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import numpy as np
import onnx
import torch

from openpi.training import config as _config

from openpi_thor._schema import ArtifactBundle
from openpi_thor._schema import EngineProfile
from openpi_thor._schema import ExportOptions
from openpi_thor.calibration import LeRobotPi05CalibrationSource
from openpi_thor.calibration import _infer_device
from openpi_thor.calibration import _prepare_policy_example
from openpi_thor.calibration import build_calibration_batches
from openpi_thor.calibration import sample_dataset_examples
from openpi_thor.engine import _build_trtexec_command
from openpi_thor.export import export_to_onnx_bundle
from openpi_thor.export import patch_model_for_export
from openpi_thor.export import prepare_model_for_export_precision
from openpi_thor.export import quantize_model
from openpi_thor.runtime import _binding_dtypes
from openpi_thor.runtime import _prepare_trt_inputs
from openpi_thor.runtime import load_pytorch_bundle
from openpi_thor.validate import compare_backends
from openpi_thor.validate import compare_tensorrt_engines
from openpi_thor.validate import _artifact_precision_from_path


@dataclasses.dataclass(frozen=True)
class _StageSpec:
    """Map a logical debug stage to exported ONNX node names."""

    name: str
    node_name: str
    qdq_prefix: str | None


_STAGE_SPECS: tuple[_StageSpec, ...] = (
    _StageSpec("input_layernorm", "input_layernorm/Cast_2", "input_layernorm"),
    _StageSpec("q_proj", "self_attn/q_proj/Add", "self_attn/q_proj"),
    _StageSpec("k_proj", "self_attn/k_proj/Add", "self_attn/k_proj"),
    _StageSpec("v_proj", "self_attn/v_proj/Add", "self_attn/v_proj"),
    _StageSpec("attention_logits", "self_attn/MatMul", "self_attn"),
    _StageSpec("attention_probs", "self_attn/Softmax", "self_attn"),
    _StageSpec("attention_output", "self_attn/MatMul_1", "self_attn"),
    _StageSpec("o_proj", "self_attn/o_proj/Add", "self_attn/o_proj"),
    _StageSpec("gate_proj", "mlp/gate_proj/MatMul", "mlp/gate_proj"),
    _StageSpec("up_proj", "mlp/up_proj/MatMul", "mlp/up_proj"),
    _StageSpec("down_proj", "mlp/down_proj/MatMul", "mlp/down_proj"),
    _StageSpec("layer_output", "Add_1", None),
    # These cache labels are heuristic but still useful for spotting where the
    # exported prefix path begins to diverge inside TensorRT.
    _StageSpec("key_cache", "self_attn/Concat_10", "self_attn"),
    _StageSpec("value_cache", "self_attn/Concat_11", "self_attn"),
)

_TORCH_MODULE_STAGE_SPECS: tuple[tuple[str, str], ...] = (
    ("input_layernorm", "paligemma_with_expert.paligemma.model.language_model.layers.{layer}.input_layernorm"),
    ("q_proj", "paligemma_with_expert.paligemma.model.language_model.layers.{layer}.self_attn.q_proj"),
    ("k_proj", "paligemma_with_expert.paligemma.model.language_model.layers.{layer}.self_attn.k_proj"),
    ("v_proj", "paligemma_with_expert.paligemma.model.language_model.layers.{layer}.self_attn.v_proj"),
    ("o_proj", "paligemma_with_expert.paligemma.model.language_model.layers.{layer}.self_attn.o_proj"),
    ("gate_proj", "paligemma_with_expert.paligemma.model.language_model.layers.{layer}.mlp.gate_proj"),
    ("up_proj", "paligemma_with_expert.paligemma.model.language_model.layers.{layer}.mlp.up_proj"),
    ("down_proj", "paligemma_with_expert.paligemma.model.language_model.layers.{layer}.mlp.down_proj"),
    ("layer_output", "paligemma_with_expert.paligemma.model.language_model.layers.{layer}"),
)


def _resolve_train_config(config: str | _config.TrainConfig) -> _config.TrainConfig:
    """Resolve a config name into the registered OpenPI train config."""

    if isinstance(config, _config.TrainConfig):
        return config
    return _config.get_config(config)


def _resolve_bundle(bundle_dir: str | Path, *, config_name: str | None = None) -> ArtifactBundle:
    """Load an existing bundle or create an empty manifest for debug work."""

    bundle_path = Path(bundle_dir).expanduser().resolve()
    metadata_path = bundle_path / "openpi_thor_bundle.json"
    if metadata_path.exists():
        return ArtifactBundle.load(bundle_path)
    if config_name is None:
        raise FileNotFoundError(f"No openpi_thor_bundle.json found in {bundle_path}")
    bundle = ArtifactBundle(bundle_dir=bundle_path, config_name=config_name)
    bundle.save()
    return bundle


def _tensor_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    """Compute compact drift metrics for one tensor pair."""

    reference = np.asarray(reference, dtype=np.float32)
    candidate = np.asarray(candidate, dtype=np.float32)
    diff = candidate - reference
    reference_flat = reference.reshape(-1)
    candidate_flat = candidate.reshape(-1)
    denom = float(np.linalg.norm(reference_flat) * np.linalg.norm(candidate_flat))
    cosine = 1.0 if denom == 0.0 else float(np.dot(reference_flat, candidate_flat) / denom)
    reference_l2 = float(np.linalg.norm(reference_flat))
    diff_l2 = float(np.linalg.norm(diff.reshape(-1)))
    return {
        "cosine": cosine,
        "mean_abs_error": float(np.mean(np.abs(diff))),
        "max_abs_error": float(np.max(np.abs(diff))),
        "reference_l2": reference_l2,
        "diff_l2": diff_l2,
        "relative_l2": 0.0 if reference_l2 == 0.0 else diff_l2 / reference_l2,
    }


def _layer_stage_order(layer: int, stage: str) -> tuple[int, int]:
    stage_index = {spec.name: index for index, spec in enumerate(_STAGE_SPECS)}
    return layer, stage_index[stage]


def _selected_stage_outputs(onnx_path: Path, *, layers: range) -> dict[str, str]:
    """Find exported tensor names for the requested early Gemma layer stages."""

    model = onnx.load(str(onnx_path), load_external_data=False)
    selected: dict[str, str] = {}
    for layer in layers:
        for spec in _STAGE_SPECS:
            expected_name = f"/layers.{layer}/{spec.node_name}"
            for node in model.graph.node:
                if node.name == expected_name:
                    selected[f"layer{layer}:{spec.name}"] = node.output[0]
                    break
    return selected


def _selected_output_value_infos(onnx_path: Path, *, selected_outputs: dict[str, str]) -> dict[str, onnx.ValueInfoProto]:
    """Infer value_info metadata for debug outputs that are not model outputs yet."""

    model = onnx.load(str(onnx_path), load_external_data=False)
    inferred_model = onnx.shape_inference.infer_shapes(model, data_prop=False)
    infos = {}
    for value_info in list(inferred_model.graph.value_info) + list(inferred_model.graph.output):
        if value_info.name in selected_outputs.values():
            infos[value_info.name] = value_info
    return infos


def _write_debug_onnx(
    src_onnx_path: Path,
    dst_onnx_path: Path,
    *,
    selected_outputs: dict[str, str],
    value_infos: dict[str, onnx.ValueInfoProto],
) -> None:
    """Append intermediate tensors as extra outputs while keeping the full graph intact."""

    dst_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(src_onnx_path), load_external_data=True)
    model_value_info_names = {value_info.name for value_info in model.graph.value_info}
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    node_output_names = {
        output_name
        for node in model.graph.node
        for output_name in node.output
        if output_name
    }
    missing_tensor_names = [
        tensor_name
        for tensor_name in selected_outputs.values()
        if tensor_name not in node_output_names
        and tensor_name not in initializer_names
        and tensor_name not in {value.name for value in model.graph.input}
    ]
    if missing_tensor_names:
        raise RuntimeError(
            f"Unable to mark debug outputs in {src_onnx_path}: missing tensors {missing_tensor_names[:5]}"
        )

    for tensor_name, value_info in value_infos.items():
        if tensor_name in model_value_info_names:
            continue
        model.graph.value_info.append(copy.deepcopy(value_info))

    existing_output_names = {value.name for value in model.graph.output}
    for tensor_name in selected_outputs.values():
        if tensor_name in existing_output_names:
            continue
        if tensor_name in value_infos:
            model.graph.output.append(copy.deepcopy(value_infos[tensor_name]))
        else:
            model.graph.output.append(onnx.helper.make_empty_tensor_value_info(tensor_name))

    onnx.external_data_helper.convert_model_to_external_data(
        model,
        all_tensors_to_one_file=True,
        location=dst_onnx_path.with_suffix(".data").name,
    )
    onnx.save(model, str(dst_onnx_path))


def _build_debug_engine(
    train_config: _config.TrainConfig,
    bundle: ArtifactBundle,
    *,
    onnx_path: Path,
    engine_path: Path,
) -> None:
    """Build a strongly typed TensorRT engine for an augmented debug ONNX model."""

    if engine_path.exists() and engine_path.stat().st_mtime >= onnx_path.stat().st_mtime:
        return
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    command = _build_trtexec_command(
        train_config,
        bundle,
        onnx_path,
        engine_path,
        EngineProfile(strongly_typed=True),
    )
    env = os.environ.copy()
    env.setdefault("APPORT_DISABLE", "1")
    subprocess.run(command, check=True, env=env, timeout=1800)


def _prepare_debug_observation(
    config: str | _config.TrainConfig,
    bundle_dir: str | Path,
    *,
    dataset_repo_id: str | None,
    dataset_root: str | Path | None,
):
    """Load one real dataset example and prepare the policy-format observation/noise pair."""

    train_config = _resolve_train_config(config)
    policy, _ = load_pytorch_bundle(train_config, bundle_dir)
    example = sample_dataset_examples(
        train_config,
        num_examples=1,
        dataset_repo_id=dataset_repo_id,
        dataset_root=dataset_root,
    )[0]
    resolved_device = _infer_device(None)
    observation, noise = _prepare_policy_example(policy, example, device=resolved_device, noise_seed=0)
    return example, observation, noise


def _run_engine_outputs(engine_path: Path, observation, noise: torch.Tensor) -> dict[str, np.ndarray]:
    """Execute one TensorRT engine and collect all named outputs on CPU."""

    from openpi_thor import trt_torch

    engine = trt_torch.Engine(str(engine_path))
    dtypes = _binding_dtypes(engine)
    bindings = _prepare_trt_inputs(observation, device="cuda", dtypes=dtypes)
    runtime_noise = noise
    if runtime_noise.ndim == 2:
        runtime_noise = runtime_noise[None, ...]
    bindings["noise"] = runtime_noise.to(device="cuda", dtype=dtypes["noise"]).contiguous()
    for name, tensor in bindings.items():
        engine.set_runtime_tensor_shape(name, tensor.shape)
    outputs = engine(**bindings)
    return {name: tensor.detach().cpu().numpy() for name, tensor in outputs.items()}


def _qdq_summary(onnx_path: Path, *, layers: range) -> dict[str, Any]:
    """Summarize QuantizeLinear/DequantizeLinear placement for early Gemma blocks."""

    model = onnx.load(str(onnx_path), load_external_data=False)
    per_layer: dict[str, dict[str, Any]] = {}
    first_qdq_difference = None
    for layer in layers:
        layer_summary: dict[str, Any] = {}
        for spec in _STAGE_SPECS:
            if spec.qdq_prefix is None:
                continue
            prefix = f"/layers.{layer}/{spec.qdq_prefix}".rstrip("/")
            qdq_ops: dict[str, int] = {}
            for node in model.graph.node:
                if prefix not in node.name:
                    continue
                if "QuantizeLinear" not in node.op_type and "DequantizeLinear" not in node.op_type:
                    continue
                qdq_ops[node.op_type] = qdq_ops.get(node.op_type, 0) + 1
            if qdq_ops:
                layer_summary[spec.name] = {"qdq_ops": qdq_ops}
        per_layer[f"layer{layer}"] = layer_summary
    return {"layers": per_layer}


def _first_qdq_difference(fp8_summary: dict[str, Any], nvfp4_summary: dict[str, Any], *, layers: range) -> dict[str, Any] | None:
    """Return the first early stage whose QDQ placement differs between fp8 and NVFP4."""

    for layer in layers:
        layer_key = f"layer{layer}"
        fp8_layers = fp8_summary["layers"].get(layer_key, {})
        nvfp4_layers = nvfp4_summary["layers"].get(layer_key, {})
        for spec in _STAGE_SPECS:
            fp8_ops = fp8_layers.get(spec.name, {}).get("qdq_ops", {})
            nvfp4_ops = nvfp4_layers.get(spec.name, {}).get("qdq_ops", {})
            if fp8_ops != nvfp4_ops:
                return {
                    "layer": layer,
                    "stage": spec.name,
                    "fp8_qdq_ops": fp8_ops,
                    "nvfp4_qdq_ops": nvfp4_ops,
                }
    return None


def _module_from_dotted_name(model: torch.nn.Module, dotted_name: str) -> torch.nn.Module | None:
    """Resolve one dotted module path relative to a root model."""

    module: torch.nn.Module = model
    for part in dotted_name.split("."):
        if not hasattr(module, part):
            return None
        module = getattr(module, part)
        if not isinstance(module, torch.nn.Module):
            return None
    return module


def _module_output_to_tensor(output: Any) -> torch.Tensor:
    """Normalize common hook outputs down to one representative tensor."""

    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item
    raise TypeError(f"Unsupported hooked output type: {type(output)!r}")


def _prepare_quantized_debug_model(
    train_config: _config.TrainConfig,
    bundle: ArtifactBundle,
    *,
    enable_llm_nvfp4: bool,
    quantize_attention_matmul: bool,
    num_calibration_samples: int,
    dataset_repo_id: str | None,
    dataset_root: str | Path | None,
) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    """Build one quantized PyTorch model that mirrors the export-time fp8/NVFP4 path."""

    policy, _ = load_pytorch_bundle(train_config, bundle.bundle_dir, pytorch_device="cuda")
    model = policy._model  # noqa: SLF001
    model.eval()
    model = patch_model_for_export(model, compute_dtype=torch.float16)
    model = prepare_model_for_export_precision(model, compute_dtype=torch.float16)
    calibration_source = LeRobotPi05CalibrationSource(
        config=train_config,
        num_samples=num_calibration_samples,
        dataset_repo_id=dataset_repo_id,
        dataset_root=dataset_root,
    )
    calibration_batches = build_calibration_batches(
        calibration_source,
        policy,
        train_config,
        device=next(model.parameters()).device,
    )
    quantized_model = quantize_model(
        model,
        calibration_data=calibration_batches,
        num_steps=bundle.num_steps or 10,
        enable_llm_nvfp4=enable_llm_nvfp4,
        quantize_attention_matmul=quantize_attention_matmul,
    )
    return quantized_model, policy._input_transform, {
        "calibration_source": calibration_source.name,
        "calibration_num_samples": len(calibration_batches),
    }


def _run_quantized_torch_stage_debug(
    train_config: _config.TrainConfig,
    bundle: ArtifactBundle,
    *,
    example: dict[str, Any],
    layers: range,
    material_relative_l2_threshold: float,
    dataset_repo_id: str | None,
    dataset_root: str | Path | None,
) -> dict[str, Any]:
    """Compare fp8 and fp8+NVFP4 inside the quantized PyTorch prefix language-model modules."""

    fp8_artifact = bundle.artifacts.get("fp8")
    nvfp4_artifact = bundle.artifacts.get("fp8_nvfp4")
    calibration_num_samples = max(
        fp8_artifact.calibration_num_samples if fp8_artifact and fp8_artifact.calibration_num_samples else 0,
        nvfp4_artifact.calibration_num_samples if nvfp4_artifact and nvfp4_artifact.calibration_num_samples else 0,
    )
    if calibration_num_samples == 0:
        calibration_num_samples = 32
    fp8_model, fp8_input_transform, fp8_meta = _prepare_quantized_debug_model(
        train_config,
        bundle,
        enable_llm_nvfp4=False,
        quantize_attention_matmul=False,
        num_calibration_samples=calibration_num_samples,
        dataset_repo_id=dataset_repo_id,
        dataset_root=dataset_root,
    )
    nvfp4_model, nvfp4_input_transform, nvfp4_meta = _prepare_quantized_debug_model(
        train_config,
        bundle,
        enable_llm_nvfp4=True,
        quantize_attention_matmul=True,
        num_calibration_samples=calibration_num_samples,
        dataset_repo_id=dataset_repo_id,
        dataset_root=dataset_root,
    )

    def capture_stage_outputs(model: torch.nn.Module, input_transform) -> tuple[dict[str, np.ndarray], torch.Tensor]:
        stage_outputs: dict[str, np.ndarray] = {}
        hooks = []

        for layer in layers:
            for stage_name, module_path_template in _TORCH_MODULE_STAGE_SPECS:
                module_path = module_path_template.format(layer=layer)
                module = _module_from_dotted_name(model, module_path)
                if module is None:
                    continue
                stage_key = f"layer{layer}:{stage_name}"

                def _hook(_module, _inputs, output, *, stage_key=stage_key):
                    tensor = _module_output_to_tensor(output)
                    stage_outputs[stage_key] = tensor.detach().float().cpu().numpy()

                hooks.append(module.register_forward_hook(_hook))

        policy_stub = type("_PolicyStub", (), {"_input_transform": input_transform, "_model": model})()
        device = next(model.parameters()).device
        observation, noise = _prepare_policy_example(policy_stub, example, device=device, noise_seed=0)
        with torch.no_grad():
            final_actions = model.sample_actions(str(device), observation, noise=noise, num_steps=bundle.num_steps or 10)
        for hook in hooks:
            hook.remove()
        return stage_outputs, final_actions.detach().float().cpu()

    fp8_outputs, fp8_actions = capture_stage_outputs(fp8_model, fp8_input_transform)
    nvfp4_outputs, nvfp4_actions = capture_stage_outputs(nvfp4_model, nvfp4_input_transform)

    stage_results: list[dict[str, Any]] = []
    first_material_difference = None
    shared_stage_keys = sorted(
        set(fp8_outputs) & set(nvfp4_outputs),
        key=lambda key: _layer_stage_order(int(key.split(":")[0].removeprefix("layer")), key.split(":")[1]),
    )
    for stage_key in shared_stage_keys:
        metrics = _tensor_metrics(fp8_outputs[stage_key], nvfp4_outputs[stage_key])
        layer_str, stage = stage_key.split(":")
        stage_result = {
            "layer": int(layer_str.removeprefix("layer")),
            "stage": stage,
            **metrics,
        }
        stage_results.append(stage_result)
        if first_material_difference is None and metrics["relative_l2"] > material_relative_l2_threshold:
            first_material_difference = dict(stage_result)
            break

    final_action_metrics = _tensor_metrics(fp8_actions.numpy(), nvfp4_actions.numpy())
    return {
        "calibration_num_samples": calibration_num_samples,
        "fp8_metadata": fp8_meta,
        "fp8_nvfp4_metadata": nvfp4_meta,
        "first_material_difference": first_material_difference,
        "stage_results": stage_results,
        "final_action_metrics": final_action_metrics,
        "notes": [
            "This compares quantized PyTorch models before ONNX/TensorRT lowering.",
            "Attention-logit and cache internals are correlated through ONNX QDQ inspection rather than direct PyTorch hooks.",
        ],
    }


def run_fp8_nvfp4_debug(
    config: str | _config.TrainConfig,
    bundle_dir: str | Path,
    *,
    dataset_repo_id: str | None = None,
    dataset_root: str | Path | None = None,
    layers: range = range(5),
    material_relative_l2_threshold: float = 0.05,
    compare_with_tensorrt: bool = False,
) -> Path:
    """Compare fp8 and fp8+NVFP4 internal layer tensors and write a debug report."""

    train_config = _resolve_train_config(config)
    bundle = _resolve_bundle(bundle_dir, config_name=train_config.name)
    fp8_onnx_path = Path(bundle.onnx_paths["fp8"]).expanduser().resolve()
    fp8_nvfp4_onnx_path = Path(bundle.onnx_paths["fp8_nvfp4"]).expanduser().resolve()
    if not fp8_onnx_path.exists() or not fp8_nvfp4_onnx_path.exists():
        raise FileNotFoundError("Both fp8 and fp8_nvfp4 ONNX artifacts are required before running the debug harness.")

    fp8_selected_outputs = _selected_stage_outputs(fp8_onnx_path, layers=layers)
    nvfp4_selected_outputs = _selected_stage_outputs(fp8_nvfp4_onnx_path, layers=layers)
    if not fp8_selected_outputs:
        raise RuntimeError(f"No /layers.<n>/ debug outputs were found in {fp8_onnx_path}")
    if not nvfp4_selected_outputs:
        raise RuntimeError(f"No /layers.<n>/ debug outputs were found in {fp8_nvfp4_onnx_path}")
    shared_output_keys = [key for key in fp8_selected_outputs if key in nvfp4_selected_outputs]
    if not shared_output_keys:
        raise RuntimeError("No shared fp8/fp8_nvfp4 debug outputs were found for the selected layers.")
    fp8_selected_outputs = {key: fp8_selected_outputs[key] for key in shared_output_keys}
    nvfp4_selected_outputs = {key: nvfp4_selected_outputs[key] for key in shared_output_keys}
    value_infos = _selected_output_value_infos(fp8_onnx_path, selected_outputs=fp8_selected_outputs)

    example, observation, noise = _prepare_debug_observation(
        train_config,
        bundle.bundle_dir,
        dataset_repo_id=dataset_repo_id,
        dataset_root=dataset_root,
    )
    torch_debug = _run_quantized_torch_stage_debug(
        train_config,
        bundle,
        example=example,
        layers=layers,
        material_relative_l2_threshold=material_relative_l2_threshold,
        dataset_repo_id=dataset_repo_id,
        dataset_root=dataset_root,
    )

    tensorrt_debug: dict[str, Any] | None = None
    if compare_with_tensorrt:
        debug_dir = bundle.bundle_dir / "debug"
        stage_results: list[dict[str, Any]] = []
        first_material_difference = None
        first_build_failure = None
        ordered_keys = sorted(
            shared_output_keys,
            key=lambda key: _layer_stage_order(int(key.split(":")[0].removeprefix("layer")), key.split(":")[1]),
        )
        for key in ordered_keys:
            layer_str, stage = key.split(":")
            layer = int(layer_str.removeprefix("layer"))
            stage_payload: dict[str, Any] = {"layer": layer, "stage": stage}
            selected_fp8 = {key: fp8_selected_outputs[key]}
            selected_nvfp4 = {key: nvfp4_selected_outputs[key]}
            fp8_debug_onnx = debug_dir / f"model_fp8_debug_{layer}_{stage}.onnx"
            nvfp4_debug_onnx = debug_dir / f"model_fp8_nvfp4_debug_{layer}_{stage}.onnx"
            fp8_debug_engine = debug_dir / f"model_fp8_debug_{layer}_{stage}.engine"
            nvfp4_debug_engine = debug_dir / f"model_fp8_nvfp4_debug_{layer}_{stage}.engine"
            _write_debug_onnx(fp8_onnx_path, fp8_debug_onnx, selected_outputs=selected_fp8, value_infos=value_infos)
            _write_debug_onnx(
                fp8_nvfp4_onnx_path,
                nvfp4_debug_onnx,
                selected_outputs=selected_nvfp4,
                value_infos=value_infos,
            )
            stage_payload.update(
                {
                    "fp8_debug_onnx": str(fp8_debug_onnx),
                    "fp8_nvfp4_debug_onnx": str(nvfp4_debug_onnx),
                    "fp8_debug_engine": str(fp8_debug_engine),
                    "fp8_nvfp4_debug_engine": str(nvfp4_debug_engine),
                }
            )
            try:
                _build_debug_engine(train_config, bundle, onnx_path=fp8_debug_onnx, engine_path=fp8_debug_engine)
                _build_debug_engine(train_config, bundle, onnx_path=nvfp4_debug_onnx, engine_path=nvfp4_debug_engine)
            except Exception as exc:  # noqa: BLE001
                stage_payload["build_error"] = str(exc)
                stage_results.append(stage_payload)
                if first_build_failure is None:
                    first_build_failure = dict(stage_payload)
                break

            fp8_outputs = _run_engine_outputs(fp8_debug_engine, observation, noise)
            nvfp4_outputs = _run_engine_outputs(nvfp4_debug_engine, observation, noise)
            metrics = _tensor_metrics(fp8_outputs[fp8_selected_outputs[key]], nvfp4_outputs[nvfp4_selected_outputs[key]])
            stage_payload.update(
                {
                    "fp8_tensor_name": fp8_selected_outputs[key],
                    "fp8_nvfp4_tensor_name": nvfp4_selected_outputs[key],
                    **metrics,
                }
            )
            stage_results.append(stage_payload)
            if first_material_difference is None and metrics["relative_l2"] > material_relative_l2_threshold:
                first_material_difference = {
                    "layer": layer,
                    "stage": stage,
                    **metrics,
                }
                break
        tensorrt_debug = {
            "first_material_difference": first_material_difference,
            "first_build_failure": first_build_failure,
            "stage_results": stage_results,
            "ranked_drift": [result for result in stage_results if "relative_l2" in result],
        }

    fp8_qdq_summary = _qdq_summary(fp8_onnx_path, layers=layers)
    nvfp4_qdq_summary = _qdq_summary(fp8_nvfp4_onnx_path, layers=layers)
    payload = {
        "phase": "debug_fp8_nvfp4",
        "bundle_dir": str(bundle.bundle_dir),
        "fp8_onnx_path": str(fp8_onnx_path),
        "fp8_nvfp4_onnx_path": str(fp8_nvfp4_onnx_path),
        "dataset_index": int(np.asarray(example["dataset_index"]).item()),
        "selected_outputs": {
            key: {
                "fp8": fp8_selected_outputs[key],
                "fp8_nvfp4": nvfp4_selected_outputs[key],
            }
            for key in shared_output_keys
        },
        "material_relative_l2_threshold": material_relative_l2_threshold,
        "torch_debug": torch_debug,
        "tensorrt_debug": tensorrt_debug,
        "graph_qdq_summary": {
            "fp8": fp8_qdq_summary,
            "fp8_nvfp4": nvfp4_qdq_summary,
            "first_qdq_difference": _first_qdq_difference(fp8_qdq_summary, nvfp4_qdq_summary, layers=layers),
        },
        "notes": [
            "This helper compares fp8 and fp8_nvfp4 directly so NVFP4 regressions can be isolated from the JAX baseline.",
            "torch_debug runs by default and isolates quantization drift before ONNX/TensorRT lowering.",
            "tensorrt_debug is optional because augmented NVFP4 stage-output engines are much slower and less stable to build.",
        ],
    }
    report_path = bundle.write_report("debug_fp8_nvfp4", payload, report_key="debug:fp8_nvfp4")
    bundle.save()
    return report_path


def _initialize_sweep_bundle(source_bundle: ArtifactBundle, target_bundle_dir: Path) -> None:
    """Create a temporary sweep bundle by symlinking the heavy shared assets."""

    target_bundle_dir.mkdir(parents=True, exist_ok=True)
    source_weight = source_bundle.weight_path
    target_weight = target_bundle_dir / source_weight.name
    if not target_weight.exists():
        os.symlink(source_weight, target_weight)
    if source_bundle.assets_dir.exists():
        target_assets = target_bundle_dir / "assets"
        if not target_assets.exists():
            os.symlink(source_bundle.assets_dir, target_assets, target_is_directory=True)


@contextlib.contextmanager
def _temporary_sweep_bundle(source_bundle: ArtifactBundle, *, suffix: str):
    """Yield a temporary per-run bundle directory for calibration sweeps."""

    with tempfile.TemporaryDirectory(prefix=f"{source_bundle.bundle_dir.name}_{suffix}_", dir=source_bundle.bundle_dir.parent) as tmpdir:
        bundle_dir = Path(tmpdir)
        _initialize_sweep_bundle(source_bundle, bundle_dir)
        yield bundle_dir


def run_fp8_calibration_sweep(
    config: str | _config.TrainConfig,
    bundle_dir: str | Path,
    *,
    reference_checkpoint_dir: str | Path,
    sample_counts: tuple[int, ...] = (32, 128, 256),
    validation_num_examples: int = 8,
    dataset_repo_id: str | None = None,
    dataset_root: str | Path | None = None,
) -> Path:
    """Sweep fp8 calibration counts and aggregate JAX/fp16 comparison reports."""

    train_config = _resolve_train_config(config)
    bundle = _resolve_bundle(bundle_dir, config_name=train_config.name)
    fp16_engine_path = bundle.get_recommended_engine_path()
    if fp16_engine_path is None:
        raise FileNotFoundError("A recommended fp16 TensorRT engine is required before running the calibration sweep.")

    examples = sample_dataset_examples(
        train_config,
        num_examples=validation_num_examples,
        dataset_repo_id=dataset_repo_id,
        dataset_root=dataset_root,
    )

    variants = (
        {
            "artifact_key": "fp8",
            "precision": "fp8",
            "enable_llm_nvfp4": False,
            "quantize_attention_matmul": False,
        },
        {
            "artifact_key": "fp8_nvfp4",
            "precision": "fp8",
            "enable_llm_nvfp4": True,
            "quantize_attention_matmul": True,
        },
    )
    runs: list[dict[str, Any]] = []
    fp16_artifact = bundle.artifacts.get("fp16")
    num_steps = fp16_artifact.num_steps if fp16_artifact and fp16_artifact.num_steps is not None else (bundle.num_steps or 10)
    for sample_count in sample_counts:
        for variant in variants:
            suffix = f"{variant['artifact_key']}_{sample_count}"
            with _temporary_sweep_bundle(bundle, suffix=suffix) as temp_bundle_dir:
                temp_bundle = export_to_onnx_bundle(
                    train_config,
                    temp_bundle_dir,
                    options=ExportOptions(
                        precision=variant["precision"],
                        num_steps=num_steps,
                        enable_llm_nvfp4=variant["enable_llm_nvfp4"],
                        quantize_attention_matmul=variant["quantize_attention_matmul"],
                        num_calibration_samples=sample_count,
                        allow_dummy_calibration=False,
                    ),
                    dataset_repo_id=dataset_repo_id,
                    dataset_root=dataset_root,
                )
                onnx_path = Path(temp_bundle.onnx_paths[variant["artifact_key"]])
                from openpi_thor.engine import build_engine

                built_bundle = build_engine(
                    train_config,
                    temp_bundle_dir,
                    onnx_path=onnx_path,
                    profile=EngineProfile(strongly_typed=True),
                )
                engine_path = Path(built_bundle.engine_paths[onnx_path.stem])
                jax_report = compare_backends(
                    train_config,
                    temp_bundle_dir,
                    examples=examples,
                    reference_checkpoint_dir=reference_checkpoint_dir,
                    candidate_backend="tensorrt",
                    engine_path=engine_path,
                    dataset_repo_id=dataset_repo_id,
                    dataset_root=dataset_root,
                )
                fp16_report = compare_tensorrt_engines(
                    train_config,
                    temp_bundle_dir,
                    candidate_engine_path=engine_path,
                    reference_engine_path=fp16_engine_path,
                    examples=examples,
                    dataset_repo_id=dataset_repo_id,
                    dataset_root=dataset_root,
                )
                temp_bundle = ArtifactBundle.load(temp_bundle_dir)
                export_report_path = temp_bundle.resolve_report_path(temp_bundle.artifacts[variant["artifact_key"]].report_paths["export"])
                export_report = json.loads(export_report_path.read_text())
                runs.append(
                    {
                        "artifact_key": variant["artifact_key"],
                        "sample_count": sample_count,
                        "onnx_path": str(onnx_path),
                        "engine_path": str(engine_path),
                        "onnx_checker_warning": export_report.get("onnx_checker_warning"),
                        "jax_report": jax_report.to_dict(),
                        "fp16_report": fp16_report.to_dict(),
                    }
                )

    payload = {
        "phase": "fp8_calibration_sweep",
        "bundle_dir": str(bundle.bundle_dir),
        "reference_checkpoint_dir": str(Path(reference_checkpoint_dir).expanduser().resolve()),
        "reference_fp16_engine": str(fp16_engine_path),
        "dataset_indices": [int(np.asarray(example["dataset_index"]).item()) for example in examples],
        "sample_counts": list(sample_counts),
        "validation_num_examples": validation_num_examples,
        "dataset_repo_id": dataset_repo_id,
        "dataset_root": str(dataset_root) if dataset_root is not None else None,
        "runs": runs,
    }
    report_path = bundle.write_report("fp8_calibration_sweep", payload, report_key="sweep:fp8_calibration")
    bundle.save()
    return report_path
