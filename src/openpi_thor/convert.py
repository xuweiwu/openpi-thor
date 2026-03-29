from __future__ import annotations

import dataclasses
import importlib.util
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
from openpi.training import config as _config

from openpi_thor._schema import ArtifactBundle
from openpi_thor._schema import default_bundle
from openpi_thor.compat import prepare_runtime

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _resolve_train_config(config: str | _config.TrainConfig) -> _config.TrainConfig:
    if isinstance(config, _config.TrainConfig):
        return config
    return _config.get_config(config)


def _load_converter_module():
    module_path = _repo_root() / "examples" / "convert_jax_model_to_pytorch.py"
    spec = importlib.util.spec_from_file_location("openpi_thor_converter_impl", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load converter module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_assets_source(checkpoint_dir: Path) -> Path | None:
    candidates = [
        checkpoint_dir / "assets",
        checkpoint_dir.parent / "assets",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def inspect_jax_checkpoint(
    checkpoint_dir: str | Path,
    *,
    restore_precision: str | None = "float32",
) -> dict[str, list[int]]:
    """Inspect a JAX checkpoint and return flattened parameter shapes.

    This is a lightweight helper for sanity-checking the source checkpoint
    before conversion without materializing a PyTorch bundle.
    """

    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    params = _model.restore_params(
        checkpoint_dir / "params",
        restore_type=np.ndarray,
        dtype=restore_precision,
    )
    flat = flax.traverse_util.flatten_dict(params, sep="/")
    return {key: [int(dim) for dim in value.shape] for key, value in flat.items()}


def convert_jax_checkpoint(
    config: str | _config.TrainConfig,
    checkpoint_dir: str | Path,
    bundle_dir: str | Path,
    *,
    precision: str = "bfloat16",
    copy_assets: bool = True,
    overwrite: bool = False,
) -> ArtifactBundle:
    """Convert a JAX checkpoint into a PyTorch bundle directory.

    The resulting bundle contains converted weights, copied assets when
    available, and an updated manifest/report entry for the conversion phase.
    """

    prepare_runtime()
    train_config = _resolve_train_config(config)
    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    bundle_dir = Path(bundle_dir).expanduser().resolve()

    if bundle_dir.exists() and any(bundle_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty bundle directory: {bundle_dir}")

    bundle_dir.mkdir(parents=True, exist_ok=True)

    converter = _load_converter_module()
    model_config = train_config.model
    if hasattr(model_config, "pytorch_compile_mode"):
        model_config = dataclasses.replace(model_config, pytorch_compile_mode=None)

    logger.info("Converting JAX checkpoint %s into PyTorch bundle %s", checkpoint_dir, bundle_dir)
    converter.convert_pi0_checkpoint(
        str(checkpoint_dir),
        precision,
        str(bundle_dir),
        model_config,
    )

    if copy_assets and not (bundle_dir / "assets").exists():
        if assets_source := _resolve_assets_source(checkpoint_dir):
            shutil.copytree(assets_source, bundle_dir / "assets", dirs_exist_ok=True)

    config_json = bundle_dir / "config.json"
    if config_json.exists():
        config_payload = json.loads(config_json.read_text())
    else:
        config_payload = {}

    bundle = default_bundle(bundle_dir, train_config.name)
    bundle.source_checkpoint_dir = str(checkpoint_dir)
    bundle.extra.update(
        {
            "converted_from": "jax",
            "pytorch_precision": precision,
        }
    )
    bundle.write_report(
        "convert_jax_to_pytorch",
        {
            "phase": "convert_jax_to_pytorch",
            "config_name": train_config.name,
            "source_checkpoint_dir": str(checkpoint_dir),
            "bundle_dir": str(bundle_dir),
            "precision": precision,
            "copy_assets": copy_assets,
            "assets_dir": str(bundle_dir / "assets") if (bundle_dir / "assets").exists() else None,
            "weight_path": str(bundle_dir / "model.safetensors"),
            "config_json": config_payload,
        },
    )
    bundle.save()
    return bundle
