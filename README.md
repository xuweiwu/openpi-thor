# openpi-thor

Jetson AGX Thor deployment helpers for OpenPI `pi05_*` models.

`openpi-thor` is designed to live at `packages/openpi-thor` inside an `openpi` checkout. Its
goal is to provide a non-Docker Jetson AGX Thor path for:

- JAX checkpoint to PyTorch bundle conversion
- ONNX export
- TensorRT engine build
- JAX/PyTorch/TensorRT validation
- WebSocket-compatible serving

The canonical user-facing interface is the installed console script:

```bash
openpi-thor ...
```

If you intentionally do not install the package entrypoint into the runtime venv, the
fallback is:

```bash
python -m openpi_thor.cli ...
```

In v1, the supported external-repo model is a companion repo cloned into an upstream-like
`openpi` fork at exactly `packages/openpi-thor`. Wheel-only installation or arbitrary checkout
paths are not supported yet.

## Compatibility assumptions

`openpi-thor` assumes:

- Jetson AGX Thor host with JetPack-managed CUDA/TensorRT
- OpenPI repo checkout available locally
- TensorRT and `trtexec` provided by the host runtime, not by this package
- Real export/build/serve work happens in a Jetson AGX Thor runtime venv, not the repo's default dev env

`openpi-thor doctor` should pass before running real export/build/serve commands.

## Tested host

The package is intended to be portable across Jetson AGX Thor setups, but the current flow
was tested on:

- Ubuntu `24.04.4 LTS`
- JetPack `7.0-b128`
- L4T `38.2.2`

## Tested dependency matrix

The exact dependency mix matters on Jetson AGX Thor. The currently tested stack for this
package is:

### Required for FP16

- PyTorch: `2.11.0`
- JAX / jaxlib: `0.5.3`
- Orbax Checkpoint: `0.11.13`
- Transformers: `4.53.2`
- ONNX: `1.18.0`
- ONNX GraphSurgeon: `0.5.8`
- TensorRT Python: `10.13.3.x`

### Required for FP8 / NVFP4

- `nvidia-modelopt==0.33.1`
- `onnxruntime>=1.24.4`

### Important notes

- `openpi-thor` uses the official `lerobot` `v0.5.0` tag by default.
- The current dataset and calibration code paths are also compatible with official
  `lerobot` `v0.4.4`.
- `lerobot` is installed separately with `--no-deps` because both official tags publish
  dependency metadata that pins `torch<2.11.0`, which can downgrade the working Jetson AGX
  Thor PyTorch stack.
- `openpi-thor` expects host TensorRT and `trtexec`; it does not install those itself.
- `openpi-thor` builds TensorRT engines with `--strongly-typed` by default. This preserves
  explicit FP32 stability islands in Gemma-based models, such as RMSNorm, and avoids the
  large numerical drift seen with weakly typed builds on Jetson AGX Thor.

## Acknowledgements

This package builds on the official Jetson AI Lab OpenPI-on-Thor tutorial:

- https://www.jetson-ai-lab.com/tutorials/openpi_on_thor/

In particular:

- `src/openpi_thor/trt_torch.py` re-homes the tutorial's `trt_torch.py` helper so
  `openpi-thor` can serve TensorRT engines without depending on an `openpi_on_thor/`
  directory at runtime
- parts of the ONNX/TensorRT compatibility flow were developed with the tutorial scripts as a
  reference, then adapted for the companion-package workflow and user-finetuned `pi05_*`
  models

## License

`openpi-thor` is licensed under Apache License 2.0. See `LICENSE`.

## Repo-level `pyproject.toml` changes

`openpi-thor` is a workspace package, so a few root-level `uv` settings in the host repo's
`pyproject.toml` matter for the Jetson AGX Thor flow:

- `tool.uv.sources`
  Makes `uv` treat `openpi`, `openpi-client`, and the official tagged `lerobot` source as
  workspace or source-controlled dependencies instead of unrelated packages.
- `tool.uv.override-dependencies`
  Pins `ml-dtypes` and `tensorstore` to versions that work with the tested JAX/ONNX stack.
- `tool.uv.conflicts`
  Prevents `uv` from trying to resolve the repo's default development environment together
  with the Jetson AGX Thor-specific `openpi-thor` dependency groups, which would otherwise
  pull incompatible PyTorch variants into one environment.

You usually do not need to edit those entries by hand, but they are part of why the setup
steps below work reliably.

## Acknowledgements

`openpi-thor` builds on the official Jetson AI Lab OpenPI-on-Thor tutorial:

- https://www.jetson-ai-lab.com/tutorials/openpi_on_thor/

In particular:

- the local TensorRT helper in `src/openpi_thor/trt_torch.py` was re-homed from the tutorial's
  `trt_torch.py` so this package no longer needs the tutorial directory at runtime
- parts of the ONNX/TensorRT compatibility work were developed with the tutorial scripts as a
  reference, then adapted for a companion-package workflow and user-finetuned `pi05_*` models
  inside upstream-like `openpi` forks

## Using `openpi-thor` from another `openpi` fork

The supported v1 companion-repo flow is:

1. clone `openpi-thor` into the host repo at exactly `packages/openpi-thor`
2. patch the host root `pyproject.toml`
3. create the Jetson AGX Thor runtime venv
4. install the local editables and Jetson AGX Thor dependency groups

This assumes an upstream-like host repo layout, including:

- `src/openpi`
- `examples/convert_jax_model_to_pytorch.py`
- `src/openpi/models_pytorch/transformers_replace/...`

Preview the required host-root changes first:

```bash
python packages/openpi-thor/scripts/patch_host_openpi.py --host-root .
```

Apply them only when the preview looks correct:

```bash
python packages/openpi-thor/scripts/patch_host_openpi.py --host-root . --write
```

The patch script only edits the host root `pyproject.toml`. It ensures:

- `tool.uv.override-dependencies` contains `ml-dtypes==0.5.1` and `tensorstore==0.1.74`
- `tool.uv.conflicts` contains the Thor dependency-group conflict entries required by `openpi-thor`
- `tool.uv.sources.lerobot` points at the official `lerobot` `v0.5.0` tag

Manual fallback:

- add the same `tool.uv.override-dependencies`, `tool.uv.conflicts`, and `tool.uv.sources.lerobot`
  entries by hand if you do not want to run the patch script

`openpi-thor` no longer requires the tutorial-only `openpi_on_thor/trt_torch.py` runtime file
to serve TensorRT engines. The package ships its own TensorRT helper now.

## Setup

The recommended non-Docker flow is:

1. create a dedicated Jetson AGX Thor venv
2. install the local repo packages without dependency resolution, so `openpi-thor` is
   available as a console script
3. install the Jetson AGX Thor runtime dependency groups with `uv`
4. verify the runtime with `openpi-thor doctor`

Example:

```bash
cd <OPENPI_REPO>

uv venv \
  --no-project \
  --no-managed-python \
  --python /usr/bin/python3 \
  --system-site-packages \
  .venv-thor

source .venv-thor/bin/activate

uv pip install --python .venv-thor/bin/python --no-deps \
  -e . \
  -e packages/openpi-client \
  -e packages/openpi-thor

uv pip install --python .venv-thor/bin/python \
  --project packages/openpi-thor \
  --group thor-pytorch \
  --group thor-runtime \
  --group thor-fp8

uv pip install --python .venv-thor/bin/python --no-deps \
  "lerobot @ git+https://github.com/huggingface/lerobot@v0.5.0"
```

These steps are intentionally split instead of using one `uv pip install` command:

- the editable repo packages are installed with `--no-deps` so they do not pull the repo's
  default `torch==2.7.1` requirement into the Jetson AGX Thor runtime
- the main Jetson AGX Thor dependency groups can be installed together
- `lerobot` is installed separately with `--no-deps` because its official dependency
  metadata can downgrade the working PyTorch stack

If you intentionally need the older official `lerobot` `v0.4.4` tag, `openpi-thor`'s
current dataset and calibration path is also compatible with it. Keep the same `--no-deps`
installation pattern:

```bash
uv pip install --python .venv-thor/bin/python --no-deps \
  "lerobot @ git+https://github.com/huggingface/lerobot@v0.4.4"
```

Make sure `trtexec` is on `PATH` before building engines:

```bash
export PATH=/usr/src/tensorrt/bin:$PATH
```

Verify the runtime:

```bash
openpi-thor doctor
```

If you do not want to install the local repo packages into the Jetson AGX Thor venv, you
can fall back to `python -m openpi_thor.cli ...`, but then you must also provide repo-local
imports through `PYTHONPATH`.

## Bundle directories

A bundle directory is the working directory for one model across the whole deployment flow.
It is usually created by `convert-jax`, and then reused by the later commands.

`--bundle-dir` should point to that directory, not to the original JAX checkpoint directory.

Typical bundle contents are:

- `model.safetensors` for the converted PyTorch weights
- `assets/` for copied normalization and config assets
- `onnx/` for exported ONNX models
- `engine/` for built TensorRT engines
- `reports/` for phase-specific JSON reports
- `openpi_thor_bundle.json` for the thin manifest that points to the latest artifacts and reports

`openpi_thor_bundle.json` is meant to stay compact. Detailed conversion, export, build, and
validation results are written into separate files under `reports/`.

## Main commands

- `openpi-thor doctor`
  Checks whether the Jetson AGX Thor runtime has the required Python packages and host tools.
- `openpi-thor convert-jax`
  Converts a JAX checkpoint into a PyTorch bundle directory.
- `openpi-thor export-onnx`
  Exports an ONNX model into an existing bundle directory.
- `openpi-thor build-engine`
  Builds a TensorRT engine from the ONNX artifact stored in the bundle. Strongly typed builds
  are the default.
- `openpi-thor prepare-engine`
  High-level command that can export ONNX, build TensorRT, and optionally validate in one
  step. Strongly typed engine builds are the default here too.
- `openpi-thor validate`
  Compares PyTorch or TensorRT outputs against the JAX reference model on real dataset
  samples.
- `openpi-thor status`
  Prints a compact summary of the bundle contents, recommended engine, validation state, and
  report file paths.
- `openpi-thor serve`
  Starts the websocket inference server for the selected or recommended engine.

Examples:

```bash
openpi-thor doctor

openpi-thor convert-jax \
  --config <PI05_TRAIN_CONFIG> \
  --checkpoint-dir /path/to/jax-checkpoint \
  --output-dir /path/to/bundle

openpi-thor export-onnx \
  --config <PI05_TRAIN_CONFIG> \
  --bundle-dir /path/to/bundle \
  --precision fp16

openpi-thor build-engine \
  --config <PI05_TRAIN_CONFIG> \
  --bundle-dir /path/to/bundle

openpi-thor validate \
  --config <PI05_TRAIN_CONFIG> \
  --bundle-dir /path/to/bundle \
  --candidate-backend pytorch

openpi-thor serve \
  --config <PI05_TRAIN_CONFIG> \
  --bundle-dir /path/to/bundle
```

Replace `<PI05_TRAIN_CONFIG>` with the config name registered in your host `openpi` fork,
for example one of your own `pi05_*` training configs.

## Recommended end-to-end flow

Use `prepare-engine` when you want one orchestration command for export, build, and optional
validation:

```bash
openpi-thor prepare-engine \
  --config <PI05_TRAIN_CONFIG> \
  --bundle-dir /path/to/bundle \
  --precision fp16 \
  --validate \
  --reference-checkpoint-dir /path/to/jax-checkpoint
```

For FP8 with real-data calibration and validation:

```bash
openpi-thor prepare-engine \
  --config <PI05_TRAIN_CONFIG> \
  --bundle-dir /path/to/bundle \
  --precision fp8 \
  --num-calibration-samples 32 \
  --validate \
  --reference-checkpoint-dir /path/to/jax-checkpoint
```

If you want to override the dataset used for calibration and validation, pass:

- `--dataset-repo-id <HF_DATASET_REPO>`
- `--dataset-root <LOCAL_LEROBOT_ROOT>`

If omitted, `openpi-thor` uses the dataset specified by the training config.

If you want to opt out of the default strongly typed TensorRT build for debugging or A/B
comparison, pass:

- `--no-strongly-typed`

Inspect the current bundle state with:

```bash
openpi-thor status --bundle-dir /path/to/bundle
```

For a more human-readable view that also loads the latest JSON reports:

```bash
openpi-thor status --bundle-dir /path/to/bundle --verbose
```

## Serving

`openpi-thor serve` starts a websocket inference server compatible with
`openpi_client.websocket_client_policy`.

If the bundle has a recommended engine, `serve` uses it automatically:

```bash
openpi-thor serve \
  --config <PI05_TRAIN_CONFIG> \
  --bundle-dir /path/to/bundle \
  --port 8000
```

Use `--engine-path` only when you want to override the recommended engine for debugging or
comparison.

## Troubleshooting

- If `torch.cuda.is_available()` becomes `False` after an install step, re-assert the tested
  Jetson AGX Thor PyTorch group.
- Weakly typed TensorRT builds can drift badly on Gemma-based models. `openpi-thor` uses
  `--strongly-typed` by default; only disable it with `--no-strongly-typed` for debugging.
- If FP8/NVFP4 export fails, confirm that `nvidia-modelopt`, `onnxruntime`, and host
  TensorRT tools are all present in the Jetson AGX Thor runtime.
- Use the repo dev environment for unit tests, not the Jetson AGX Thor runtime venv:

```bash
uv run --package openpi-thor pytest packages/openpi-thor/tests -m "not manual"
```
