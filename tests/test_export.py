import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import onnx

from openpi_thor import calibration
from openpi_thor import export


def test_prefer_system_blackwell_ptxas_sets_env_when_system_is_newer(monkeypatch) -> None:
    monkeypatch.delenv("TRITON_PTXAS_BLACKWELL_PATH", raising=False)
    system_ptxas = Path("/tmp/system-ptxas")

    monkeypatch.setattr(export, "_system_blackwell_ptxas_candidates", lambda: [system_ptxas])
    monkeypatch.setattr(export, "_bundled_blackwell_ptxas_path", lambda: Path("/tmp/bundled-ptxas"))
    monkeypatch.setattr(
        export,
        "_read_ptxas_release",
        lambda path: (13, 0) if path == system_ptxas else (12, 9),
    )
    monkeypatch.setattr(Path, "exists", lambda self: self == system_ptxas)

    selected = export._prefer_system_blackwell_ptxas()

    assert selected == str(system_ptxas)
    assert os.environ["TRITON_PTXAS_BLACKWELL_PATH"] == str(system_ptxas)


def test_prefer_system_blackwell_ptxas_keeps_existing_override(monkeypatch) -> None:
    monkeypatch.setenv("TRITON_PTXAS_BLACKWELL_PATH", "/custom/ptxas")
    monkeypatch.setattr(export, "_system_blackwell_ptxas_candidates", lambda: [])

    selected = export._prefer_system_blackwell_ptxas()

    assert selected == "/custom/ptxas"


def test_attention_mask_fill_value_is_half_safe() -> None:
    assert export._attention_mask_fill_value(torch.float16) == float(torch.finfo(torch.float16).min)
    assert export._attention_mask_fill_value(torch.float32) == -2.3819763e38


class _DummyExportModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision_tower = torch.nn.Module()
        self.vision_tower.vision_model = torch.nn.Module()
        self.vision_tower.vision_model.embeddings = torch.nn.Module()
        self.vision_tower.vision_model.embeddings.patch_embedding = torch.nn.Conv2d(3, 3, 1)
        self.vision_tower.vision_model.embeddings.position_embedding = torch.nn.Embedding(4, 3)
        self.vision_tower.vision_model.encoder = torch.nn.Module()
        self.vision_tower.vision_model.encoder.layers = torch.nn.ModuleList([torch.nn.Module()])
        self.vision_tower.vision_model.encoder.layers[0].layer_norm1 = torch.nn.LayerNorm(3)
        self.vision_tower.vision_model.encoder.layers[0].self_attn = torch.nn.Module()
        self.vision_tower.vision_model.encoder.layers[0].self_attn.q_proj = torch.nn.Linear(3, 3)
        self.vision_tower.vision_model.post_layernorm = torch.nn.LayerNorm(3)
        self.action_in_proj = torch.nn.Linear(3, 3)
        self.action_out_proj = torch.nn.Linear(3, 3)
        self.time_mlp_in = torch.nn.Linear(3, 3)
        self.time_mlp_out = torch.nn.Linear(3, 3)
        self.action_time_mlp_in = torch.nn.Linear(3, 3)
        self.action_time_mlp_out = torch.nn.Linear(3, 3)
        self.other_linear = torch.nn.Linear(3, 3)
        self.register_buffer("rotary_emb_inv_freq", torch.ones(3, dtype=torch.float32))


class _CaptureModule(torch.nn.Module):
    def __init__(self, output: torch.Tensor | None = None) -> None:
        super().__init__()
        self.output = output
        self.last_input = None

    def forward(self, x):
        self.last_input = x
        if self.output is None:
            return x
        return self.output


class _DummyGemmaMLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = _CaptureModule(output=torch.tensor([float("inf")], dtype=torch.float16))
        self.up_proj = _CaptureModule(output=torch.tensor([1.0], dtype=torch.float16))
        self.down_proj = _CaptureModule()
        self.act_fn = lambda x: x


class _DummyQuantizer:
    def __init__(self) -> None:
        self._trt_high_precision_dtype = None
        self._onnx_quantizer_type = None


class _DummyQuantizedLinear(torch.nn.Linear):
    def __init__(self) -> None:
        super().__init__(3, 3)
        self.weight_quantizer = _DummyQuantizer()


def test_prepare_model_for_export_precision_keeps_stability_modules_float32() -> None:
    model = _DummyExportModule().to(dtype=torch.float32)

    export.prepare_model_for_export_precision(model, compute_dtype=torch.float16)

    assert model.vision_tower.vision_model.embeddings.patch_embedding.weight.dtype == torch.float32
    assert model.vision_tower.vision_model.embeddings.position_embedding.weight.dtype == torch.float32
    assert model.action_in_proj.weight.dtype == torch.float32
    assert model.action_out_proj.weight.dtype == torch.float32
    assert model.time_mlp_in.weight.dtype == torch.float32
    assert model.time_mlp_out.weight.dtype == torch.float32
    assert model.action_time_mlp_in.weight.dtype == torch.float32
    assert model.action_time_mlp_out.weight.dtype == torch.float32
    assert model.other_linear.weight.dtype == torch.float16


def test_patch_model_for_export_sanitizes_nonfinite_gemma_mlp_hidden() -> None:
    model = torch.nn.Module()
    model.paligemma_with_expert = torch.nn.Module()
    model.paligemma_with_expert.paligemma = torch.nn.Module()
    model.paligemma_with_expert.paligemma.model = torch.nn.Module()
    model.paligemma_with_expert.paligemma.model.vision_tower = torch.nn.Module()
    model.paligemma_with_expert.paligemma.model.vision_tower.vision_model = torch.nn.Module()
    model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings = torch.nn.Module()
    model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding = torch.nn.Conv2d(3, 3, 1)
    model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.position_embedding = torch.nn.Embedding(4, 3)
    model.paligemma_with_expert.paligemma.model.language_model = torch.nn.Module()
    model.paligemma_with_expert.paligemma.model.language_model.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.paligemma_with_expert.paligemma.model.language_model.layers[0].mlp = _DummyGemmaMLP()

    export.patch_model_for_export(model)

    mlp = model.paligemma_with_expert.paligemma.model.language_model.layers[0].mlp
    _ = mlp(torch.tensor([1.0], dtype=torch.float16))

    assert mlp._openpi_thor_export_mlp_patch is True  # noqa: SLF001
    assert torch.isfinite(mlp.down_proj.last_input).all()


def test_apply_nvfp4_gemma_mlp_weight_only_quant_cfg_only_targets_weight_quantizer() -> None:
    quant_cfg = {"quant_cfg": {"existing": {"enable": True}}}

    updated = export._apply_nvfp4_gemma_mlp_weight_only_quant_cfg(quant_cfg)

    assert updated["quant_cfg"]["existing"] == {"enable": True}
    assert updated["quant_cfg"][export._NVFP4_GEMMA_MLP_WEIGHT_SELECTOR] == {
        "num_bits": (2, 1),
        "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
        "axis": None,
        "enable": True,
    }


def test_mark_nvfp4_gemma_mlp_weight_quantizers_only_updates_mlp_linears() -> None:
    model = torch.nn.Module()
    model.mlp = torch.nn.Module()
    model.mlp.down_proj = _DummyQuantizedLinear().to(dtype=torch.float16)
    model.self_attn = torch.nn.Module()
    model.self_attn.q_proj = _DummyQuantizedLinear().to(dtype=torch.float16)

    export._mark_nvfp4_gemma_mlp_weight_quantizers(model)

    assert model.mlp.down_proj.weight_quantizer._trt_high_precision_dtype == "Half"
    assert model.mlp.down_proj.weight_quantizer._onnx_quantizer_type == "static"
    assert model.self_attn.q_proj.weight_quantizer._trt_high_precision_dtype is None
    assert model.self_attn.q_proj.weight_quantizer._onnx_quantizer_type is None


def test_postprocess_onnx_model_preserves_sibling_artifacts(tmp_path: Path, monkeypatch) -> None:
    onnx_path = tmp_path / "current.onnx"
    onnx_path.write_text("placeholder")
    (tmp_path / "current.data").write_text("current")
    (tmp_path / "sibling.engine").write_text("engine")
    (tmp_path / "sibling.onnx").write_text("onnx")
    (tmp_path / "sibling.data").write_text("data")
    junk = tmp_path / "onnx__MatMul_1"
    junk.write_text("junk")

    fake_onnx_model = SimpleNamespace()

    monkeypatch.setattr("onnx.load", lambda *args, **kwargs: fake_onnx_model)
    monkeypatch.setattr("onnx.save", lambda *args, **kwargs: None)
    monkeypatch.setattr("onnx.external_data_helper.convert_model_to_external_data", lambda *args, **kwargs: None)

    export.postprocess_onnx_model(onnx_path)

    assert (tmp_path / "current.onnx").exists()
    assert (tmp_path / "current.data").exists()
    assert (tmp_path / "sibling.engine").exists()
    assert (tmp_path / "sibling.onnx").exists()
    assert (tmp_path / "sibling.data").exists()
    assert not junk.exists()


def test_calibration_noise_tensor_keeps_float32_on_cuda(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeGenerator:
        def __init__(self, *, device: str) -> None:
            captured["generator_device"] = device

        def manual_seed(self, seed: int):
            captured["seed"] = seed
            return self

    def _fake_randn(*shape, generator=None, device=None, dtype=None):
        captured["shape"] = shape
        captured["device"] = device
        captured["dtype"] = dtype
        return "noise"

    monkeypatch.setattr(torch, "Generator", _FakeGenerator)
    monkeypatch.setattr(torch, "randn", _fake_randn)

    noise = calibration._noise_tensor(10, 14, seed=123, device=torch.device("cuda"))

    assert noise == "noise"
    assert captured["generator_device"] == "cuda"
    assert captured["seed"] == 123
    assert captured["shape"] == (1, 10, 14)
    assert captured["device"] == torch.device("cuda")
    assert captured["dtype"] == torch.float32


def test_trt_high_precision_dtype_for_tensor_dtype_matches_expected_labels() -> None:
    assert export._trt_high_precision_dtype_for_tensor_dtype(torch.float16) == "Half"
    assert export._trt_high_precision_dtype_for_tensor_dtype(torch.bfloat16) == "BFloat16"
    assert export._trt_high_precision_dtype_for_tensor_dtype(torch.float32) == "Float"


def test_fast_float8e4m3fn_array_matches_onnx_reference() -> None:
    from onnx.reference.ops.op_cast import Cast_19 as Cast

    values = torch.tensor(
        [0.0, -0.0, 1e-9, 1e9, -1e9, float("inf"), float("-inf"), float("nan"), 0.1, 448.0, -448.0],
        dtype=torch.float32,
    ).numpy()

    reference = Cast.eval(values, to=onnx.TensorProto.FLOAT8E4M3FN)
    fast = export._fast_float8e4m3fn_array(values)

    assert reference.tobytes() == fast.tobytes()


def test_validate_exported_onnx_accepts_nvfp4_block_size_warning(monkeypatch, tmp_path: Path) -> None:
    from onnx.onnx_cpp2py_export.checker import ValidationError

    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"onnx")

    def _raise(_path: str) -> None:
        raise ValidationError(
            "Unrecognized attribute: block_size for operator DequantizeLinear"
        )

    monkeypatch.setattr(onnx.checker, "check_model", _raise)

    warning = export._validate_exported_onnx(onnx_path, enable_llm_nvfp4=True)

    assert "block_size" in warning


def test_validate_exported_onnx_raises_non_nvfp4_checker_error(monkeypatch, tmp_path: Path) -> None:
    from onnx.onnx_cpp2py_export.checker import ValidationError

    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"onnx")

    def _raise(_path: str) -> None:
        raise ValidationError("generic checker failure")

    monkeypatch.setattr(onnx.checker, "check_model", _raise)

    with pytest.raises(ValidationError):
        export._validate_exported_onnx(onnx_path, enable_llm_nvfp4=True)
