from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from openpi_thor.host_integration import doctor_host_integration_warnings
from openpi_thor.host_integration import plan_host_pyproject_patch
from openpi_thor.host_integration import write_host_pyproject_patch


def _make_host_repo(tmp_path: Path, pyproject_text: str) -> Path:
    host_root = tmp_path / "host-openpi"
    (host_root / "packages/openpi-thor").mkdir(parents=True)
    (host_root / "src/openpi/models_pytorch/transformers_replace").mkdir(parents=True)
    (host_root / "examples").mkdir(parents=True)
    (host_root / "examples/convert_jax_model_to_pytorch.py").write_text("# fixture\n")
    (host_root / "pyproject.toml").write_text(pyproject_text)
    return host_root


def test_host_patch_preview_is_non_mutating_and_preserves_unrelated_entries(tmp_path: Path) -> None:
    host_root = _make_host_repo(
        tmp_path,
        """
[project]
name = "openpi"
version = "0.1.0"

[tool.uv]
override-dependencies = ["existing==1.0"]

[tool.uv.sources]
openpi = { workspace = true }

[tool.uv.workspace]
members = ["packages/*"]

[tool.ruff]
line-length = 88
""".strip()
        + "\n",
    )
    original = (host_root / "pyproject.toml").read_text()

    plan = plan_host_pyproject_patch(host_root)

    assert not plan.errors
    assert plan.updated_text is not None
    assert (host_root / "pyproject.toml").read_text() == original
    assert "line-length = 88" in plan.updated_text
    assert 'override-dependencies = ["existing==1.0", "ml-dtypes==0.5.1", "tensorstore==0.1.74"]' in plan.updated_text
    assert 'openpi = { workspace = true }' in plan.updated_text
    assert 'lerobot = { git = "https://github.com/huggingface/lerobot", tag = "v0.5.0" }' in plan.updated_text


def test_host_patch_write_is_idempotent(tmp_path: Path) -> None:
    host_root = _make_host_repo(
        tmp_path,
        """
[project]
name = "openpi"
version = "0.1.0"

[tool.uv.workspace]
members = ["packages/*"]
""".strip()
        + "\n",
    )

    first = write_host_pyproject_patch(host_root)
    second = write_host_pyproject_patch(host_root)

    assert first.changed
    assert not first.errors
    assert not second.changed
    assert not second.errors
    assert second.already_correct


def test_host_patch_reports_missing_prerequisites(tmp_path: Path) -> None:
    host_root = tmp_path / "broken-host"
    host_root.mkdir()
    (host_root / "pyproject.toml").write_text("[project]\nname = \"openpi\"\nversion = \"0.1.0\"\n")

    plan = plan_host_pyproject_patch(host_root)

    assert plan.errors
    assert any("packages/openpi-thor" in item for item in plan.errors)


def test_patch_script_preview_and_write(tmp_path: Path) -> None:
    host_root = _make_host_repo(
        tmp_path,
        """
[project]
name = "openpi"
version = "0.1.0"

[tool.uv.workspace]
members = ["packages/*"]
""".strip()
        + "\n",
    )
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "patch_host_openpi.py"

    preview = subprocess.run(
        [sys.executable, str(script_path), "--host-root", str(host_root)],
        capture_output=True,
        text=True,
        check=False,
    )
    written_before = (host_root / "pyproject.toml").read_text()
    apply = subprocess.run(
        [sys.executable, str(script_path), "--host-root", str(host_root), "--write"],
        capture_output=True,
        text=True,
        check=False,
    )
    written_after = (host_root / "pyproject.toml").read_text()

    assert preview.returncode == 0
    assert "Preview only" in preview.stdout
    assert apply.returncode == 0
    assert written_before != written_after


def test_doctor_host_integration_warns_when_patch_is_needed(tmp_path: Path) -> None:
    host_root = _make_host_repo(
        tmp_path,
        """
[project]
name = "openpi"
version = "0.1.0"

[tool.uv.workspace]
members = ["packages/*"]
""".strip()
        + "\n",
    )
    module_file = host_root / "packages/openpi-thor/src/openpi_thor/doctor.py"
    module_file.parent.mkdir(parents=True, exist_ok=True)
    module_file.write_text("# fixture\n")

    info, warnings = doctor_host_integration_warnings(module_file)

    assert info["companion_host_root"] == str(host_root)
    assert warnings
    assert "patch_host_openpi.py" in warnings[0]
