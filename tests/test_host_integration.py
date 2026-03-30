from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from openpi_thor.host_integration import doctor_host_integration_warnings
from openpi_thor.host_integration import HOST_LEROBOT_COMPATIBILITY_FILE
from openpi_thor.host_integration import HOST_TRANSFORMS_COMPATIBILITY_FILE
from openpi_thor.host_integration import plan_host_pyproject_patch
from openpi_thor.host_integration import write_host_pyproject_patch


def _make_host_repo(tmp_path: Path, pyproject_text: str) -> Path:
    host_root = tmp_path / "host-openpi"
    (host_root / "packages/openpi-thor").mkdir(parents=True)
    (host_root / "src/openpi/models_pytorch/transformers_replace").mkdir(parents=True)
    (host_root / "src/openpi/training").mkdir(parents=True)
    (host_root / "src/openpi").mkdir(parents=True, exist_ok=True)
    (host_root / "examples").mkdir(parents=True)
    (host_root / "examples/convert_jax_model_to_pytorch.py").write_text("# fixture\n")
    (host_root / "pyproject.toml").write_text(pyproject_text)
    (host_root / HOST_LEROBOT_COMPATIBILITY_FILE).write_text(
        "import jax\n"
        "import lerobot.common.datasets.lerobot_dataset as lerobot_dataset\n"
        "import numpy as np\n"
    )
    (host_root / HOST_TRANSFORMS_COMPATIBILITY_FILE).write_text(
        "import dataclasses\n"
        "from typing import Protocol, TypeAlias, TypeVar, runtime_checkable\n\n"
        "@dataclasses.dataclass(frozen=True)\n"
        "class PromptFromLeRobotTask(DataTransformFn):\n"
        '    """Extracts a prompt from the current LeRobot dataset task."""\n\n'
        "    # Contains the LeRobot dataset tasks (dataset.meta.tasks).\n"
        "    tasks: dict[int, str]\n\n"
        "    def __call__(self, data: DataDict) -> DataDict:\n"
        '        if "task_index" not in data:\n'
        '            raise ValueError(\'Cannot extract prompt without "task_index"\')\n\n'
        '        task_index = int(data["task_index"])\n'
        "        if (prompt := self.tasks.get(task_index)) is None:\n"
        '            raise ValueError(f"{task_index=} not found in task mapping: {self.tasks}")\n\n'
        '        return {**data, "prompt": prompt}\n'
    )
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
    assert (host_root / HOST_LEROBOT_COMPATIBILITY_FILE).read_text().splitlines()[1] == (
        "import lerobot.common.datasets.lerobot_dataset as lerobot_dataset"
    )
    assert "line-length = 88" in plan.updated_text
    assert 'override-dependencies = ["existing==1.0", "ml-dtypes==0.5.1", "tensorstore==0.1.74"]' in plan.updated_text
    assert 'openpi = { workspace = true }' in plan.updated_text
    assert 'openpi-client = { workspace = true }' in plan.updated_text
    assert 'lerobot = { git = "https://github.com/huggingface/lerobot", tag = "v0.5.0" }' in plan.updated_text
    assert host_root / HOST_LEROBOT_COMPATIBILITY_FILE in plan.extra_file_updates
    assert host_root / HOST_TRANSFORMS_COMPATIBILITY_FILE in plan.extra_file_updates
    assert "try:" in plan.extra_file_updates[host_root / HOST_LEROBOT_COMPATIBILITY_FILE]
    assert "from typing import Any, Protocol, TypeAlias, TypeVar, runtime_checkable" in (
        plan.extra_file_updates[host_root / HOST_TRANSFORMS_COMPATIBILITY_FILE]
    )
    assert "# LerobotDataset v3: pandas.DataFrame" in plan.extra_file_updates[host_root / HOST_TRANSFORMS_COMPATIBILITY_FILE]


def test_host_patch_replaces_old_ml_dtypes_pin(tmp_path: Path) -> None:
    host_root = _make_host_repo(
        tmp_path,
        """
[project]
name = "openpi"
version = "0.1.0"

[tool.uv]
override-dependencies = ["ml-dtypes==0.4.1", "tensorstore==0.1.74"]

[tool.uv.workspace]
members = ["packages/*"]
""".strip()
        + "\n",
    )

    plan = plan_host_pyproject_patch(host_root)

    assert plan.updated_text is not None
    assert 'ml-dtypes==0.4.1' not in plan.updated_text
    assert 'override-dependencies = ["ml-dtypes==0.5.1", "tensorstore==0.1.74"]' in plan.updated_text


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
    assert "import lerobot.datasets.lerobot_dataset as lerobot_dataset" in (
        host_root / HOST_LEROBOT_COMPATIBILITY_FILE
    ).read_text()
    assert "# LerobotDataset v3: pandas.DataFrame" in (host_root / HOST_TRANSFORMS_COMPATIBILITY_FILE).read_text()


def test_host_patch_skips_data_loader_when_newer_lerobot_import_is_already_present(tmp_path: Path) -> None:
    host_root = _make_host_repo(
        tmp_path,
        """
[project]
name = "openpi"
version = "0.1.0"

[tool.uv]
override-dependencies = ["ml-dtypes==0.5.1", "tensorstore==0.1.74"]
conflicts = [
    [
        { package = "openpi" },
        { package = "openpi-thor", group = "thor-pytorch" },
    ],
    [
        { package = "openpi-thor" },
        { package = "openpi-thor", group = "thor-pytorch" },
    ],
    [
        { package = "openpi-thor", group = "thor-pytorch" },
        { package = "openpi-thor", group = "thor-lerobot" },
    ],
]

[tool.uv.sources]
openpi = { workspace = true }
openpi-client = { workspace = true }
lerobot = { git = "https://github.com/huggingface/lerobot", tag = "v0.5.0" }

[tool.uv.workspace]
members = ["packages/*"]
""".strip()
        + "\n",
    )
    (host_root / HOST_LEROBOT_COMPATIBILITY_FILE).write_text(
        "import jax\n"
        "import lerobot.datasets.lerobot_dataset as lerobot_dataset\n"
        "import numpy as np\n"
    )
    (host_root / HOST_TRANSFORMS_COMPATIBILITY_FILE).write_text(
        "from typing import Any\n\n"
        "@dataclasses.dataclass(frozen=True)\n"
        "class PromptFromLeRobotTask(DataTransformFn):\n"
        '    """Extracts a prompt from the current LeRobot dataset task."""\n\n'
        "    # Contains the LeRobot dataset tasks (dataset.meta.tasks).\n"
        "    # In practice this can be a dict or a pandas.DataFrame\n"
        "    tasks: Any\n"
        "    # LerobotDataset v3: pandas.DataFrame\n"
    )

    plan = plan_host_pyproject_patch(host_root)

    assert not plan.changed
    assert not plan.errors
    assert not plan.extra_file_updates
    assert any("already supports newer LeRobot versions" in item for item in plan.already_correct)
    assert any("dict and DataFrame LeRobot task metadata" in item for item in plan.already_correct)


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
    data_loader_before = (host_root / HOST_LEROBOT_COMPATIBILITY_FILE).read_text()
    transforms_before = (host_root / HOST_TRANSFORMS_COMPATIBILITY_FILE).read_text()
    apply = subprocess.run(
        [sys.executable, str(script_path), "--host-root", str(host_root), "--write"],
        capture_output=True,
        text=True,
        check=False,
    )
    written_after = (host_root / "pyproject.toml").read_text()
    data_loader_after = (host_root / HOST_LEROBOT_COMPATIBILITY_FILE).read_text()
    transforms_after = (host_root / HOST_TRANSFORMS_COMPATIBILITY_FILE).read_text()

    assert preview.returncode == 0
    assert "Preview only" in preview.stdout
    assert apply.returncode == 0
    assert written_before != written_after
    assert data_loader_before != data_loader_after
    assert transforms_before != transforms_after
    assert "import lerobot.datasets.lerobot_dataset as lerobot_dataset" in data_loader_after
    assert "# LerobotDataset v3: pandas.DataFrame" in transforms_after


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
