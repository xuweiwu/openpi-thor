import contextlib
import io
import json
import runpy
import sys
from pathlib import Path

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


def test_python_m_cli_executes_main(monkeypatch) -> None:
    monkeypatch.setattr(doctor, "run_doctor", lambda: _FakeReport())
    monkeypatch.setattr(sys, "argv", ["openpi_thor.cli", "doctor"])

    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        runpy.run_module("openpi_thor.cli", run_name="__main__")

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
        runpy.run_module("openpi_thor.cli", run_name="__main__")

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
        runpy.run_module("openpi_thor.cli", run_name="__main__")

    payload = json.loads(stdout.getvalue())
    assert payload["bundle_dir"] == str(tmp_path)
    assert payload["recommended_engine"] == "/tmp/model.engine"
    assert payload["verbose"] is True
