# openpi-thor

Jetson AGX Thor deployment companion for OpenPI `pi05_*` models.

`openpi-thor` is designed to live at `packages/openpi-thor` inside an `openpi` checkout. It
provides a non-Docker Jetson AGX Thor path for:

- JAX checkpoint to PyTorch conversion
- ONNX export
- TensorRT engine build
- JAX/PyTorch/TensorRT validation
- websocket serving

The canonical user-facing interface is:

```bash
openpi-thor ...
```

If you intentionally do not install the console script into the runtime venv, the fallback is:

```bash
python -m openpi_thor.cli ...
```

## Quick start

If you already have an `openpi` checkout on Jetson AGX Thor, this is the shortest path to a
first FP16 TensorRT engine.

### 1. Put the package in the supported location

The supported v1 companion-repo layout is:

```text
<OPENPI_REPO>/packages/openpi-thor
```

If you are using `openpi-thor` from another `openpi` fork, clone it there first:

```bash
cd <OPENPI_REPO>
git clone <OPENPI_THOR_REPO_URL> packages/openpi-thor
```

### 2. Patch the host root `pyproject.toml`

Preview the required changes:

```bash
python packages/openpi-thor/scripts/patch_host_openpi.py --host-root .
```

Apply them:

```bash
python packages/openpi-thor/scripts/patch_host_openpi.py --host-root . --write
```

### 3. Create the Jetson AGX Thor runtime venv

```bash
cd <OPENPI_REPO>

uv venv \
  --no-project \
  --no-managed-python \
  --python /usr/bin/python3 \
  --system-site-packages \
  .venv-thor

source .venv-thor/bin/activate
```

### 4. Install the package and runtime dependencies

```bash
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

Make sure `trtexec` is on `PATH`:

```bash
export PATH=/usr/src/tensorrt/bin:$PATH
```

### 5. Verify the runtime

```bash
openpi-thor doctor
```

### 6. Convert the JAX checkpoint into a bundle

`convert-jax` creates the bundle directory. The path passed as `--bundle-dir` here is the
same directory that later commands reuse as `--bundle-dir`. See
[Bundle directories](#bundle-directories) below.

```bash
openpi-thor convert-jax \
  --config <PI05_TRAIN_CONFIG> \
  --checkpoint-dir /path/to/jax-checkpoint \
  --bundle-dir /path/to/bundle
```

### 7. Export ONNX and build a TensorRT engine

`prepare-engine` is the fastest path for first use. FP16 is the safest starting point.

```bash
openpi-thor prepare-engine \
  --config <PI05_TRAIN_CONFIG> \
  --bundle-dir /path/to/bundle \
  --precision fp16
```

This builds TensorRT with strongly typed precision by default.

### 8. Inspect the result or serve it

```bash
openpi-thor status --bundle-dir /path/to/bundle --verbose
```

```bash
openpi-thor serve \
  --config <PI05_TRAIN_CONFIG> \
  --bundle-dir /path/to/bundle \
  --port 8000
```

Replace `<PI05_TRAIN_CONFIG>` with the config name registered in your host `openpi` fork.

## What `openpi-thor` assumes

- Jetson AGX Thor host with JetPack-managed CUDA and TensorRT
- an upstream-like local `openpi` checkout
- TensorRT Python bindings and `trtexec` provided by the host runtime
- real export/build/serve work happens in a dedicated Jetson AGX Thor runtime venv, not the
  repo's default dev environment

## Bundle directories

A bundle directory is the working directory for one model across the whole deployment flow.
It is usually created by `convert-jax` and then reused by later commands.

`--bundle-dir` should point to that directory, not to the original JAX checkpoint directory.

Typical contents:

- `model.safetensors`
- `assets/`
- `onnx/`
- `engine/`
- `reports/`
- `openpi_thor_bundle.json`

`openpi_thor_bundle.json` is the thin manifest. Detailed phase outputs are written into
separate JSON files under `reports/`.

## Main commands

- `openpi-thor doctor`
  Checks whether the Jetson AGX Thor runtime has the required Python packages and host tools.
- `openpi-thor convert-jax`
  Converts a JAX checkpoint into a PyTorch bundle directory.
- `openpi-thor export-onnx`
  Exports an ONNX model into an existing bundle directory.
- `openpi-thor build-engine`
  Builds a TensorRT engine from the ONNX artifact stored in the bundle.
- `openpi-thor prepare-engine`
  High-level command that can export ONNX, build TensorRT, and optionally validate in one step.
- `openpi-thor validate`
  Compares PyTorch or TensorRT outputs against the JAX reference model on real dataset samples.
- `openpi-thor status`
  Prints a compact summary of bundle contents, recommended engine, validation state, and report files.
- `openpi-thor serve`
  Starts the websocket inference server for the selected or recommended engine.

## Common workflows

### Build FP16 first

Start with FP16 before spending time on FP8 or NVFP4:

```bash
openpi-thor prepare-engine \
  --config <PI05_TRAIN_CONFIG> \
  --bundle-dir /path/to/bundle \
  --precision fp16 \
  --validate \
  --reference-checkpoint-dir /path/to/jax-checkpoint
```

### Build FP8 with real-data calibration

```bash
openpi-thor prepare-engine \
  --config <PI05_TRAIN_CONFIG> \
  --bundle-dir /path/to/bundle \
  --precision fp8 \
  --num-calibration-samples 32 \
  --validate \
  --reference-checkpoint-dir /path/to/jax-checkpoint
```

Optional dataset overrides for calibration and validation:

- `--dataset-repo-id <HF_DATASET_REPO>`
- `--dataset-root <LOCAL_LEROBOT_ROOT>`

If omitted, `openpi-thor` uses the dataset specified by the training config.

### Override the recommended engine

Normally `serve` uses the recommended engine recorded in the bundle. Use `--engine-path` only
for A/B tests or debugging.

## Companion-repo integration details

The supported v1 companion-repo model assumes:

- checkout path is exactly `packages/openpi-thor`
- the host repo looks like upstream `openpi`
- the host repo still contains:
  - `src/openpi`
  - `examples/convert_jax_model_to_pytorch.py`
  - `src/openpi/models_pytorch/transformers_replace/...`

The patch script only edits the host root `pyproject.toml`. It ensures:

- `tool.uv.override-dependencies` contains:
  - `ml-dtypes==0.5.1`
  - `tensorstore==0.1.74`
- `tool.uv.conflicts` contains the Thor dependency-group conflict entries required by `openpi-thor`
- `tool.uv.sources` contains:
  - `openpi = { workspace = true }`
  - `openpi-client = { workspace = true }`
  - `lerobot = { git = "...", tag = "v0.5.0" }`

Manual fallback:

- add those same `tool.uv.override-dependencies`, `tool.uv.conflicts`, and `tool.uv.sources`
  entries by hand if you do not want to run the patch script

## Tested host and dependency matrix

The package is intended to be portable across Jetson AGX Thor setups, but the current flow was
tested on:

- Ubuntu `24.04.4 LTS`
- JetPack `7.0-b128`
- L4T `38.2.2`

Tested stack for the FP16 path:

- PyTorch `2.11.0`
- torchvision `0.26.0`
- JAX / jaxlib `0.5.3`
- Orbax Checkpoint `0.11.13`
- Transformers `4.53.2`
- ONNX `1.18.0`
- ONNX GraphSurgeon `0.5.8`
- TensorRT Python `10.13.3.x`

Additional packages for FP8 / NVFP4:

- `nvidia-modelopt==0.33.1`
- `onnxruntime>=1.24.4`

Important notes:

- `openpi-thor` uses the official `lerobot` `v0.5.0` tag by default.
- The current dataset and calibration code paths are also compatible with official `lerobot`
  `v0.4.4`.
- `lerobot` is installed separately with `--no-deps` because both official tags publish
  dependency metadata that pins `torch<2.11.0`, which can downgrade the working Jetson AGX Thor
  PyTorch stack.
- `openpi-thor` expects host TensorRT and `trtexec`; it does not install those itself.
- `openpi-thor` builds TensorRT engines with `--strongly-typed` by default. This preserves
  explicit FP32 stability islands in Gemma-based models, such as RMSNorm, and avoids the large
  numerical drift seen with weakly typed builds on Jetson AGX Thor.

## Troubleshooting

- If `torch.cuda.is_available()` becomes `False` after an install step, re-assert the tested
  Jetson AGX Thor PyTorch group.
- If `prepare-engine` fails very early on import, check that the runtime env includes both
  `torch` and `torchvision`.
- Weakly typed TensorRT builds can drift badly on Gemma-based models. `openpi-thor` uses
  `--strongly-typed` by default; only disable it with `--no-strongly-typed` for debugging.
- If FP8/NVFP4 export fails, confirm that `nvidia-modelopt`, `onnxruntime`, and host TensorRT
  tools are all present in the Jetson AGX Thor runtime.
- If you keep a backup of this repo, do not leave another `openpi-thor` checkout under
  `packages/`, because `uv` will see duplicate workspace members with the same package name.
- Use the repo dev environment for unit tests, not the Jetson AGX Thor runtime venv:

```bash
uv run --package openpi-thor pytest packages/openpi-thor/tests -m "not manual"
```

## Acknowledgements

This package builds on the official Jetson AI Lab OpenPI-on-Thor tutorial:

- https://www.jetson-ai-lab.com/tutorials/openpi_on_thor/

In particular:

- `src/openpi_thor/trt_torch.py` re-homes the tutorial's `trt_torch.py` helper so
  `openpi-thor` can serve TensorRT engines without depending on an `openpi_on_thor/`
  directory at runtime
- parts of the ONNX/TensorRT compatibility flow were developed with the tutorial scripts as a
  reference, then adapted for the companion-package workflow and user-finetuned `pi05_*` models

## License

`openpi-thor` is licensed under Apache License 2.0. See `LICENSE`.
