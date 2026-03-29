import os
from pathlib import Path
from types import SimpleNamespace

import torch

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
