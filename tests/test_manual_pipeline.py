from pathlib import Path

import pytest

from openpi_thor._schema import ArtifactBundle
from openpi_thor.validate import compare_backends


JAX_CHECKPOINT = Path("/home/that/openpi/checkpoints/pi05_xlerobot_pinc_finetune")
PYTORCH_BUNDLE = Path("/home/that/openpi/checkpoints/pi05_xlerobot_pinc_finetune_pytorch")
FP8_ENGINE = PYTORCH_BUNDLE / "engine" / "model_fp8.engine"


@pytest.mark.manual
def test_manual_real_fixture_pytorch_matches_jax() -> None:
    if not JAX_CHECKPOINT.exists() or not PYTORCH_BUNDLE.exists():
        pytest.skip("Real fixture checkpoint(s) are not available on this host.")

    report = compare_backends(
        "pi05_xlerobot_pinc_finetune",
        PYTORCH_BUNDLE,
        reference_checkpoint_dir=JAX_CHECKPOINT,
        candidate_backend="pytorch",
        num_examples=4,
    )

    assert report.passed, report.to_dict()


@pytest.mark.manual
def test_manual_real_fixture_tensorrt_matches_jax() -> None:
    if not JAX_CHECKPOINT.exists() or not PYTORCH_BUNDLE.exists() or not FP8_ENGINE.exists():
        pytest.skip("Real fixture TensorRT artifacts are not available on this host.")

    if (PYTORCH_BUNDLE / "openpi_thor_bundle.json").exists():
        bundle = ArtifactBundle.load(PYTORCH_BUNDLE)
    else:
        bundle = ArtifactBundle(bundle_dir=PYTORCH_BUNDLE, config_name="pi05_xlerobot_pinc_finetune")
    bundle.precision = "fp8"
    bundle.set_engine_path("model_fp8", FP8_ENGINE)
    bundle.save()

    report = compare_backends(
        "pi05_xlerobot_pinc_finetune",
        PYTORCH_BUNDLE,
        reference_checkpoint_dir=JAX_CHECKPOINT,
        candidate_backend="tensorrt",
        engine_path=FP8_ENGINE,
        num_examples=2,
    )

    assert report.passed, report.to_dict()
