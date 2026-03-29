from __future__ import annotations

import importlib
import shutil
import subprocess
from pathlib import Path

import torch

from openpi_thor._schema import DoctorReport
from openpi_thor.compat import prepare_runtime
from openpi_thor.host_integration import doctor_host_integration_warnings


def _command_output(command: list[str]) -> tuple[bool, str]:
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)
    return True, (result.stdout or result.stderr).strip()


def _import_version(module_name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)
    return True, getattr(module, "__version__", "unknown")


def run_doctor() -> DoctorReport:
    errors: list[str] = []
    warnings: list[str] = []
    info: dict[str, object] = {}

    info["torch_version"] = torch.__version__
    info["torch_cuda_available"] = torch.cuda.is_available()
    if not torch.cuda.is_available():
        errors.append("PyTorch does not see a CUDA-capable device.")

    if shutil.which("trtexec") is None:
        errors.append("trtexec is not on PATH.")
    else:
        ok, output = _command_output(["trtexec", "--version"])
        if ok:
            info["trtexec_version"] = output
        else:
            warnings.append(f"Unable to query trtexec version: {output}")

    for module_name in ("tensorrt", "jax", "orbax.checkpoint", "onnx_graphsurgeon", "nvtx", "transformers"):
        ok, version = _import_version(module_name)
        key = module_name.replace(".", "_")
        if ok:
            info[key] = version
        else:
            errors.append(f"{module_name} import failed: {version}")

    ok, version = _import_version("modelopt")
    if ok:
        info["modelopt"] = version
    else:
        warnings.append(f"modelopt import failed: {version}")

    if shutil.which("nvidia-smi") is not None:
        ok, output = _command_output(["nvidia-smi", "-L"])
        if ok:
            info["nvidia_smi"] = output
        else:
            warnings.append(f"nvidia-smi check failed: {output}")

    nv_tegra_release = Path("/etc/nv_tegra_release")
    if nv_tegra_release.exists():
        info["nv_tegra_release"] = nv_tegra_release.read_text().strip()

    try:
        prepare_runtime()
        info["runtime_overlay"] = "ok"
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Runtime compatibility overlay failed: {exc}")

    companion_info, companion_warnings = doctor_host_integration_warnings(__file__)
    info.update(companion_info)
    warnings.extend(companion_warnings)

    return DoctorReport(
        passed=not errors,
        errors=errors,
        warnings=warnings,
        info=info,
    )
