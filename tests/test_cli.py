import contextlib
import io
import json
import sys
from pathlib import Path

import pytest

import openpi_thor.cli as cli
import openpi_thor.doctor as doctor
import openpi_thor.workflow as workflow


class _FakeReport:
    def to_dict(self) -> dict[str, object]:
        return {
            "passed": True,
            "errors": [],
            "warnings": [],
            "info": {"source": "test"},
        }


def _run_cli(argv: list[str], monkeypatch) -> tuple[int, str]:
    monkeypatch.setattr(sys, "argv", argv)
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        try:
            cli.main()
        except SystemExit as exc:
            code = int(exc.code or 0)
        else:
            code = 0
    return code, stdout.getvalue()


def test_cli_main_executes_doctor(monkeypatch) -> None:
    monkeypatch.setattr(doctor, "run_doctor", lambda: _FakeReport())
    monkeypatch.setattr(sys, "argv", ["openpi_thor.cli", "doctor"])

    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        cli.main()

    payload = json.loads(stdout.getvalue())
    assert payload["passed"] is True
    assert payload["info"]["source"] == "test"


def test_prepare_engine_cli_dispatches_to_workflow(monkeypatch, tmp_path: Path) -> None:
    bundle = type("Bundle", (), {"metadata_path": tmp_path / "openpi_thor_bundle.json"})()
    monkeypatch.setattr(workflow, "prepare_engine", lambda *args, **kwargs: bundle)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "openpi_thor.cli",
            "prepare-engine",
            "--config",
            "pi05_xlerobot_pinc_finetune",
            "--bundle-dir",
            str(tmp_path),
            "--validate",
            "--dataset-repo-id",
            "ambient-robots/take-part-lego-build-teleop",
        ],
    )

    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        cli.main()

    assert str(bundle.metadata_path) in stdout.getvalue()


def test_status_cli_prints_bundle_status(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        workflow,
        "bundle_status",
        lambda bundle_dir, verbose=False: {
            "bundle_dir": str(bundle_dir),
            "recommended_engine": "/tmp/model.engine",
            "verbose": verbose,
        },
    )
    monkeypatch.setattr(sys, "argv", ["openpi_thor.cli", "status", "--bundle-dir", str(tmp_path), "--verbose"])

    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        cli.main()

    payload = json.loads(stdout.getvalue())
    assert payload["bundle_dir"] == str(tmp_path)
    assert payload["recommended_engine"] == "/tmp/model.engine"
    assert payload["verbose"] is True


def test_validate_tensorrt_cli_prints_report(monkeypatch, tmp_path: Path) -> None:
    class _ValidationReport:
        def to_dict(self) -> dict[str, object]:
            return {
                "reference_backend": "tensorrt",
                "candidate_backend": "tensorrt",
                "reference_path": "/tmp/model_fp16.engine",
                "candidate_path": "/tmp/model_fp8.engine",
                "passed": False,
            }

    import openpi_thor.validate as validate

    monkeypatch.setattr(validate, "compare_tensorrt_engines", lambda *args, **kwargs: _ValidationReport())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "openpi_thor.cli",
            "validate-tensorrt",
            "--config",
            "pi05_xlerobot_pinc_finetune",
            "--bundle-dir",
            str(tmp_path),
            "--candidate-engine-path",
            "/tmp/model_fp8.engine",
        ],
    )

    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        cli.main()

    payload = json.loads(stdout.getvalue())
    assert payload["reference_backend"] == "tensorrt"
    assert payload["candidate_backend"] == "tensorrt"
    assert payload["candidate_path"] == "/tmp/model_fp8.engine"


@pytest.mark.parametrize("help_flag", ["--help", "-h"])
def test_top_level_help_lists_command_descriptions(monkeypatch, help_flag: str) -> None:
    code, output = _run_cli(["openpi_thor.cli", help_flag], monkeypatch)

    assert code == 0
    assert "doctor" in output
    assert "Check the runtime, TensorRT tools, and host" in output
    assert "prepare-engine" in output
    assert "Export ONNX, build TensorRT, and optionally validate." in output
    assert "validate-tensorrt" in output
    assert "Compare one TensorRT engine directly against another" in output


@pytest.mark.parametrize(
    ("subcommand", "help_flag", "phrases"),
    [
            (
                "doctor",
                "--help",
                [
                    "Check the runtime, TensorRT tools, and host integration.",
                ],
            ),
            (
                "doctor",
                "-h",
                [
                    "Check the runtime, TensorRT tools, and host integration.",
                ],
            ),
            (
                "convert-jax",
                "--help",
                [
                    "Convert a JAX checkpoint into a PyTorch bundle.",
                    "Registered OpenPI training config name",
                    "Bundle directory to create or update",
                ],
        ),
        (
            "export-onnx",
            "--help",
            [
                "Export an ONNX graph from an existing bundle.",
                "ONNX export precision",
                "Override the dataset repo id used for real calibration",
            ],
        ),
            (
                "build-engine",
                "--help",
                [
                    "Build a TensorRT engine from a bundle's ONNX artifact.",
                    "Optional explicit ONNX path to build from.",
                    "Preserve explicit dtypes from the ONNX graph.",
                ],
        ),
        (
            "validate",
            "--help",
            [
                "Compare PyTorch or TensorRT outputs against the JAX reference.",
                "Backend to evaluate against JAX.",
                "Specific TensorRT engine to validate",
            ],
        ),
        (
            "validate-tensorrt",
            "--help",
            [
                "Compare one TensorRT engine directly against another engine.",
                "TensorRT engine path to evaluate as the candidate.",
                "Defaults to the bundle's recommended engine.",
            ],
        ),
            (
                "prepare-engine",
                "--help",
                [
                    "Export ONNX, build TensorRT, and optionally validate.",
                    "Override the dataset repo id used for calibration and validation.",
                    "Preserve explicit dtypes from the ONNX graph during engine build.",
                ],
        ),
            (
                "status",
                "--help",
                [
                    "Show bundle contents, reports, and the recommended engine.",
                    "Also load the JSON report payloads",
                ],
            ),
            (
                "serve",
                "--help",
                [
                    "Start the websocket inference server.",
                    "Optional explicit engine path.",
                    "Refuse to serve unvalidated fp8/NVFP4 artifacts",
                ],
        ),
    ],
)
def test_subcommand_help_includes_high_signal_descriptions(
    monkeypatch,
    subcommand: str,
    help_flag: str,
    phrases: list[str],
) -> None:
    code, output = _run_cli(["openpi_thor.cli", subcommand, help_flag], monkeypatch)

    assert code == 0
    for phrase in phrases:
        assert phrase in output
