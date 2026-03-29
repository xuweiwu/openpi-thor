from pathlib import Path

from openpi.training import config as _config

from openpi_thor._schema import ArtifactBundle
from openpi_thor._schema import EngineProfile
from openpi_thor.engine import _build_trtexec_command


def test_build_trtexec_command_defaults_to_strongly_typed(monkeypatch, tmp_path: Path) -> None:
    train_config = _config.get_config("pi05_xlerobot_pinc_finetune")
    bundle = ArtifactBundle(bundle_dir=tmp_path, config_name=train_config.name, precision="fp8")

    monkeypatch.setattr(
        "openpi_thor.engine._onnx_input_names",
        lambda path: {"images", "img_masks", "lang_tokens", "lang_masks", "state", "noise"},
    )

    command = _build_trtexec_command(
        train_config,
        bundle,
        tmp_path / "model_fp8.onnx",
        tmp_path / "model_fp8.engine",
        EngineProfile(),
    )
    joined = " ".join(command)

    assert "--stronglyTyped" in command
    assert "--fp16" not in command
    assert "--fp8" not in command
    assert f"noise:1x{train_config.model.action_horizon}x{train_config.model.action_dim}" in joined
    assert f"lang_tokens:1x{train_config.model.max_token_len}" in joined


def test_build_trtexec_command_supports_weakly_typed_fp16_opt_out(monkeypatch, tmp_path: Path) -> None:
    train_config = _config.get_config("pi05_xlerobot_pinc_finetune")
    bundle = ArtifactBundle(bundle_dir=tmp_path, config_name=train_config.name, precision="fp8")

    monkeypatch.setattr(
        "openpi_thor.engine._onnx_input_names",
        lambda path: {"images", "img_masks", "lang_tokens", "lang_masks", "state", "noise"},
    )

    command = _build_trtexec_command(
        train_config,
        bundle,
        tmp_path / "model_fp16.onnx",
        tmp_path / "model_fp16.engine",
        EngineProfile(strongly_typed=False),
    )

    assert "--fp16" in command
    assert "--fp8" not in command


def test_build_trtexec_command_supports_weakly_typed_fp8_opt_out(monkeypatch, tmp_path: Path) -> None:
    train_config = _config.get_config("pi05_xlerobot_pinc_finetune")
    bundle = ArtifactBundle(bundle_dir=tmp_path, config_name=train_config.name, precision="fp8")

    monkeypatch.setattr(
        "openpi_thor.engine._onnx_input_names",
        lambda path: {"images", "img_masks", "lang_tokens", "lang_masks", "state", "noise"},
    )

    command = _build_trtexec_command(
        train_config,
        bundle,
        tmp_path / "model_fp8.onnx",
        tmp_path / "model_fp8.engine",
        EngineProfile(strongly_typed=False),
    )

    assert "--stronglyTyped" not in command
    assert "--fp16" in command
    assert "--fp8" in command


def test_build_trtexec_command_supports_explicit_strongly_typed(monkeypatch, tmp_path: Path) -> None:
    train_config = _config.get_config("pi05_xlerobot_pinc_finetune")
    bundle = ArtifactBundle(bundle_dir=tmp_path, config_name=train_config.name, precision="fp16")

    monkeypatch.setattr(
        "openpi_thor.engine._onnx_input_names",
        lambda path: {"images", "img_masks", "lang_tokens", "lang_masks", "state", "noise"},
    )

    command = _build_trtexec_command(
        train_config,
        bundle,
        tmp_path / "model_fp16.onnx",
        tmp_path / "model_fp16.engine",
        EngineProfile(strongly_typed=True),
    )

    assert "--stronglyTyped" in command
    assert "--fp16" not in command
    assert "--fp8" not in command
