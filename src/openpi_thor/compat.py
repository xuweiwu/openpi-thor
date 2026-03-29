from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import sys
from typing import Any


_PREPARED = False


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _replacement_root() -> Path:
    return _repo_root() / "src" / "openpi" / "models_pytorch" / "transformers_replace" / "models"


def _module_sources() -> dict[str, Path]:
    root = _replacement_root()
    return {
        "transformers.models.gemma.configuration_gemma": root / "gemma" / "configuration_gemma.py",
        "transformers.models.gemma.modeling_gemma": root / "gemma" / "modeling_gemma.py",
        "transformers.models.paligemma.modeling_paligemma": root / "paligemma" / "modeling_paligemma.py",
        "transformers.models.siglip.modeling_siglip": root / "siglip" / "modeling_siglip.py",
        "transformers.models.siglip.check": root / "siglip" / "check.py",
    }


def _patched_source(module_name: str, path: Path) -> str:
    source = path.read_text()
    if module_name == "transformers.models.gemma.modeling_gemma":
        source = source.replace(
            '    def extra_repr(self):\n        repr_str = f"{tuple(self.weight.shape)}, eps={self.eps}"',
            "    def extra_repr(self):\n"
            "        if hasattr(self, 'weight') and self.weight is not None:\n"
            '            repr_str = f"{tuple(self.weight.shape)}, eps={self.eps}"\n'
            "        else:\n"
            '            repr_str = f"eps={self.eps}"',
        )
        source = source.replace(
            "attn_output = attn_output.reshape(*input_shape, -1).contiguous()",
            "attn_output = attn_output.reshape(*input_shape, self.config.num_attention_heads * self.head_dim).contiguous()",
        )
    return source


def _load_overlay_module(module_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_loader(module_name, loader=None, origin=str(path))
    if spec is None:
        raise ImportError(f"Unable to create spec for {module_name}")
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(path)
    sys.modules[module_name] = module
    exec(compile(_patched_source(module_name, path), str(path), "exec"), module.__dict__)
    return module


def _install_torch_onnx_compat_aliases() -> None:
    """Restore private ONNX module paths expected by ModelOpt.

    Torch 2.11 moved ``torch.onnx._globals`` under the internal
    ``torchscript_exporter`` package. NVIDIA ModelOpt still imports the older
    private path, so we alias it in-memory instead of patching site-packages.
    """

    import torch.onnx
    import torch.onnx._internal

    alias_map = {
        "torch.onnx._globals": "torch.onnx._internal.torchscript_exporter._globals",
        "torch.onnx._type_utils": "torch.onnx._internal.torchscript_exporter._type_utils",
        "torch.onnx._internal.jit_utils": "torch.onnx._internal.torchscript_exporter.jit_utils",
    }

    for legacy_name, target_name in alias_map.items():
        if legacy_name in sys.modules:
            continue
        try:
            module = importlib.import_module(target_name)
        except ImportError:
            continue
        sys.modules[legacy_name] = module
        if legacy_name.startswith("torch.onnx._internal."):
            setattr(torch.onnx._internal, legacy_name.rsplit(".", 1)[-1], module)
        else:
            setattr(torch.onnx, legacy_name.rsplit(".", 1)[-1], module)

    try:
        symbolic_public = importlib.import_module("torch.onnx.symbolic_opset14")
        symbolic_internal = importlib.import_module("torch.onnx._internal.torchscript_exporter.symbolic_opset14")
    except ImportError:
        return

    for helper_name in ("_attention_scale", "_causal_attention_mask"):
        if not hasattr(symbolic_public, helper_name) and hasattr(symbolic_internal, helper_name):
            setattr(symbolic_public, helper_name, getattr(symbolic_internal, helper_name))


def prepare_runtime() -> None:
    """Install the vendored Transformers replacements in memory.

    The Jetson tutorial copies these files into site-packages. The Thor package keeps
    site-packages untouched and overlays only the modules we need.
    """

    global _PREPARED  # noqa: PLW0603
    if _PREPARED:
        return

    import transformers

    _install_torch_onnx_compat_aliases()

    sources = _module_sources()
    loaded: dict[str, Any] = {}
    for module_name, path in sources.items():
        parent_name, leaf_name = module_name.rsplit(".", 1)
        parent_module = importlib.import_module(parent_name)
        module = _load_overlay_module(module_name, path)
        setattr(parent_module, leaf_name, module)
        loaded[module_name] = module

    gemma_module = loaded["transformers.models.gemma.modeling_gemma"]
    paligemma_module = loaded["transformers.models.paligemma.modeling_paligemma"]
    siglip_check_module = loaded["transformers.models.siglip.check"]

    transformers.GemmaForCausalLM = gemma_module.GemmaForCausalLM
    transformers.PaliGemmaForConditionalGeneration = paligemma_module.PaliGemmaForConditionalGeneration
    transformers.models.gemma.GemmaForCausalLM = gemma_module.GemmaForCausalLM
    transformers.models.paligemma.PaliGemmaForConditionalGeneration = paligemma_module.PaliGemmaForConditionalGeneration
    transformers.models.siglip.check = siglip_check_module

    _PREPARED = True
