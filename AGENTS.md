# AGENTS.md

Guidance for future coding agents working inside `packages/openpi-thor`.

## Scope

This package is a Jetson AGX Thor deployment companion for OpenPI `pi05_*` models. It is
not a standalone wheel-first project yet. The supported v1 integration model is:

- clone `openpi-thor` into a host `openpi` checkout at exactly `packages/openpi-thor`
- patch the host repo for companion integration
- create a dedicated Jetson AGX Thor runtime venv
- use the installed `openpi-thor` console script inside that runtime

Do not assume arbitrary checkout paths, Docker-first workflows, or wheel-only installation.

## Architecture

Core surfaces:

- `cli.py`
  public commands such as `doctor`, `convert-jax`, `prepare-engine`, `status`, `serve`
- `workflow.py`
  orchestration for export/build/validate flows
- `_schema.py`
  bundle manifest and report model
- `runtime.py`
  PyTorch and TensorRT-backed policy loading
- `export.py`
  ONNX export and quantization hooks
- `engine.py`
  `trtexec` command construction and engine build metadata
- `validate.py`
  JAX/PyTorch/TensorRT backend comparisons
- `host_integration.py`
  host repo patching and companion checkout checks
- `trt_torch.py`
  local TensorRT helper re-homed from the Jetson tutorial so runtime no longer depends on
  `openpi_on_thor`

## Non-obvious rules

### Companion checkout path matters

The supported source path is:

- `packages/openpi-thor`

`compat.py` and `convert.py` still rely on an upstream-like host repo layout. That is
intentional in v1. Do not casually generalize those path assumptions without replacing them
with a cleaner host-discovery layer.

### Do not reintroduce runtime dependence on `openpi_on_thor`

`openpi-thor` used to depend on `openpi_on_thor/trt_torch.py`. That dependency was removed.
Future runtime or serving code should keep using:

- `src/openpi_thor/trt_torch.py`

The tutorial directory may exist on some hosts, but it is reference-only.

### Strongly typed TensorRT is the safe default

For Gemma-based `pi05_*` models on Jetson AGX Thor, weakly typed TensorRT builds can drift
badly because TensorRT can collapse intended FP32 stability islands into FP16. In particular,
Gemma RMSNorm overflow caused severe output errors until `--stronglyTyped` was used.

Default behavior should remain:

- strongly typed TensorRT builds on by default
- weakly typed builds only as an explicit debugging opt-out

### Current public NVFP4 policy is attention-only

The public `--enable-llm-nvfp4` path is no longer the old Gemma MLP weight-only fallback.
After the internal fp8/NVFP4 investigation, the current shipped behavior is:

- NVFP4 across all Gemma attention layers
- explicit attention-matmul quantization on
- Gemma MLP kept on fp8

Do not casually reintroduce either of these older public-path ideas:

- broad attention+MLP full-layer NVFP4
- Gemma MLP weight-only NVFP4

The first one regressed badly after TensorRT lowering. The second one stayed closer in accuracy
but was slower than plain fp8 because TensorRT lowered it into cast/dequant-heavy kernels.

### TensorRT profiles matter for NVFP4 decisions

For fp8/NVFP4 work, backend quality is not just about validation metrics. TensorRT layer profiles
were the key signal in the NVFP4 investigation:

- good candidates avoided `ReplCastMulCast`-style dominance
- good candidates preserved fused attention kernels on the hot path
- bad candidates could look reasonable in quantized PyTorch and still regress only after
  ONNX/TensorRT lowering

If you change the NVFP4 path, check all three:

- JAX-referenced validation
- fp8-engine comparison
- `trtexec` layer profiles

### `state` being unused in `pi05` TensorRT graphs is expected

For `pi05` models, `state` may appear unused or only shape-used in the ONNX graph. This is
not automatically a bug. Do not misdiagnose that as the source of TensorRT drift without
checking the actual model path first.

### Bundle manifest vs reports

Keep:

- `openpi_thor_bundle.json` as the thin manifest
- `reports/*.json` for detailed phase outputs

Do not move detailed validation/debug payloads back into the manifest unless there is a very
good reason.

## Fresh setup that actually worked

Use a dedicated Jetson AGX Thor runtime venv, not the repo dev `.venv`.

Tested flow:

```bash
uv venv \
  --no-project \
  --no-managed-python \
  --python /usr/bin/python3 \
  --system-site-packages \
  .venv-thor

uv pip install --python .venv-thor/bin/python --no-deps \
  -e . \
  -e packages/openpi-client \
  -e packages/openpi-thor

uv pip install --python .venv-thor/bin/python \
  --project packages/openpi-thor \
  --group thor-pytorch \
  --group thor-runtime

uv pip install --python .venv-thor/bin/python --no-deps \
  "lerobot @ git+https://github.com/huggingface/lerobot@v0.5.0"
```

Also needed:

```bash
export PATH=/usr/src/tensorrt/bin:$PATH
```

Notes:

- `lerobot` must stay on `--no-deps`
- `lerobot v0.5.0` is the default source tag
- code paths used here were also checked against official `v0.4.4`
- `modelopt` is optional for FP16, required for FP8/NVFP4

## Host `pyproject.toml` expectations

The host repo needs:

- `tool.uv.override-dependencies`
  - `ml-dtypes==0.5.1`
  - `tensorstore==0.1.74`
- `tool.uv.conflicts`
  - the Thor dependency-group conflict entries
- `tool.uv.sources`
  - `openpi = { workspace = true }`
  - `openpi-client = { workspace = true }`
  - `lerobot = { git = "...", tag = "v0.5.0" }`

Use:

```bash
python packages/openpi-thor/scripts/patch_host_openpi.py --host-root .
python packages/openpi-thor/scripts/patch_host_openpi.py --host-root . --write
```

The patch script should stay:

- preview-first by default
- idempotent
- limited to the small set of host files that `openpi-thor` depends on

Today that means:

- the host root `pyproject.toml`
- `src/openpi/training/data_loader.py`
- `src/openpi/transforms.py`

The two source patches matter for newer `lerobot` tags:

- `data_loader.py`
  needs the newer `lerobot.datasets.lerobot_dataset` import path, with a fallback for older tags
- `transforms.py`
  needs the newer `PromptFromLeRobotTask` logic so `dataset_meta.tasks` can be either a dict or
  a pandas DataFrame, and the patch must also add `Any` to the typing import when needed

## Clean-room issues already discovered

These are real problems found by rebuilding from a clean companion checkout:

1. Host patching must replace stale `ml-dtypes` pins instead of appending conflicting ones.
2. Host patching must add `tool.uv.sources.openpi = { workspace = true }`.
3. `thor-pytorch` must include `torchvision==0.26.0` because `prepare-engine` imports
   `calibration.py`, which imports `torchvision`.
4. A backup checkout with the same package name under `packages/` breaks workspace resolution.
   If a user keeps a backup, it must live outside `packages/`.

## Runtime and debug lessons

- `openpi-thor doctor` can still pass even if `trtexec --version` exits nonzero, as long as
  `trtexec` exists on `PATH`.
- JAX may fall back to CPU if the runtime lacks a CUDA-enabled `jaxlib`. Conversion still
  works, but it is slower.
- Strongly typed TensorRT engine builds can take many minutes. During build, the target engine
  file may remain size `0` until `trtexec` finishes writing it.
- Use `openpi-thor status --verbose` to inspect phase reports after long-running commands.
- For fp8/NVFP4, do not assume more calibration samples help. On the tested stack, plain fp8
  did not improve when calibration increased from `32` to `128`, so `32` remains the default.
- The internal NVFP4 sweep in `debug_nvfp4.py` exists for a reason. Keep it internal, but do not
  rip it out unless the public path becomes simple enough that the extra profiling/debug harness is
  no longer needed.

## Preferred validation order

When debugging runtime quality, use this order:

1. `convert-jax`
2. `validate --candidate-backend pytorch`
3. `export-onnx --precision fp16`
4. `build-engine` with strongly typed default
5. `validate --candidate-backend tensorrt`
6. only then explore FP8 or NVFP4

If FP16 TensorRT is wrong, do not blame calibration first.

## Good development habits here

- When you touch host integration, test the patch script on a fixture repo and on the real
  host checkout shape.
- When you touch runtime dependencies, test from a fresh Jetson AGX Thor venv, not only from
  the repo dev environment.
- When you touch bundle metadata, check both:
  - the thin manifest
  - the detailed `reports/*.json`
- Prefer adding a focused regression test for every companion-workflow issue you fix.

## Quick commands

Preview host patch:

```bash
python packages/openpi-thor/scripts/patch_host_openpi.py --host-root .
```

Runtime check:

```bash
export PATH=/usr/src/tensorrt/bin:$PATH
.venv-thor/bin/openpi-thor doctor
```

Real bundle flow:

```bash
.venv-thor/bin/openpi-thor convert-jax \
  --config <PI05_TRAIN_CONFIG> \
  --checkpoint-dir /path/to/jax-checkpoint \
  --bundle-dir /path/to/bundle

.venv-thor/bin/openpi-thor prepare-engine \
  --config <PI05_TRAIN_CONFIG> \
  --bundle-dir /path/to/bundle \
  --precision fp16
```
