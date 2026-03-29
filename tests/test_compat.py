from pathlib import Path
import sys
import types

from openpi_thor import compat


def test_patched_source_applies_gemma_fixes(tmp_path: Path) -> None:
    source_path = tmp_path / "modeling_gemma.py"
    source_path.write_text(
        "    def extra_repr(self):\n"
        '        repr_str = f"{tuple(self.weight.shape)}, eps={self.eps}"\n'
        "        return repr_str\n"
        "attn_output = attn_output.reshape(*input_shape, -1).contiguous()\n"
    )

    patched = compat._patched_source("transformers.models.gemma.modeling_gemma", source_path)

    assert "if hasattr(self, 'weight')" in patched
    assert "self.config.num_attention_heads * self.head_dim" in patched


def test_patched_source_leaves_other_modules_unchanged(tmp_path: Path) -> None:
    source_path = tmp_path / "other.py"
    source_path.write_text("value = 1\n")

    assert compat._patched_source("transformers.models.siglip.check", source_path) == "value = 1\n"


def test_install_torch_onnx_compat_aliases_reuses_internal_globals(monkeypatch) -> None:
    legacy_names = ["torch.onnx._globals", "torch.onnx._type_utils", "torch.onnx._internal.jit_utils"]
    internal_names = {
        "torch.onnx._internal.torchscript_exporter._globals": types.SimpleNamespace(GLOBALS=object()),
        "torch.onnx._internal.torchscript_exporter._type_utils": types.SimpleNamespace(JitScalarType=object()),
        "torch.onnx._internal.torchscript_exporter.jit_utils": types.SimpleNamespace(GraphContext=object()),
        "torch.onnx._internal.torchscript_exporter.symbolic_opset14": types.SimpleNamespace(
            _attention_scale=object(),
            _causal_attention_mask=object(),
        ),
    }
    legacy_before = {name: sys.modules.pop(name, None) for name in legacy_names}
    public_symbolic = types.SimpleNamespace()
    real_import_module = compat.importlib.import_module

    def fake_import_module(name: str, package: str | None = None):
        if name in internal_names:
            return internal_names[name]
        if name == "torch.onnx.symbolic_opset14":
            return public_symbolic
        return real_import_module(name, package)

    monkeypatch.setattr(compat.importlib, "import_module", fake_import_module)

    compat._install_torch_onnx_compat_aliases()

    assert sys.modules["torch.onnx._globals"] is internal_names["torch.onnx._internal.torchscript_exporter._globals"]
    assert sys.modules["torch.onnx._type_utils"] is internal_names["torch.onnx._internal.torchscript_exporter._type_utils"]
    assert sys.modules["torch.onnx._internal.jit_utils"] is internal_names["torch.onnx._internal.torchscript_exporter.jit_utils"]
    import torch.onnx
    import torch.onnx._internal
    assert torch.onnx._globals is internal_names["torch.onnx._internal.torchscript_exporter._globals"]
    assert torch.onnx._type_utils is internal_names["torch.onnx._internal.torchscript_exporter._type_utils"]
    assert torch.onnx._internal.jit_utils is internal_names["torch.onnx._internal.torchscript_exporter.jit_utils"]
    assert public_symbolic._attention_scale is internal_names["torch.onnx._internal.torchscript_exporter.symbolic_opset14"]._attention_scale
    assert public_symbolic._causal_attention_mask is internal_names["torch.onnx._internal.torchscript_exporter.symbolic_opset14"]._causal_attention_mask

    for name, module in legacy_before.items():
        sys.modules.pop(name, None)
        if module is not None:
            sys.modules[name] = module
