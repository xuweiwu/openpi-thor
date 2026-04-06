# openpi-thor development

Use this skill when working on `packages/openpi-thor`, especially for:

- companion-repo integration
- Jetson AGX Thor runtime setup
- ONNX / TensorRT export flows
- bundle metadata and reporting changes

## First checks

1. Confirm the checkout path is exactly `packages/openpi-thor`.
2. Check whether the host repo companion patch is applied:
   ```bash
   python packages/openpi-thor/scripts/patch_host_openpi.py --host-root .
   ```
3. Make sure `trtexec` is on `PATH`:
   ```bash
   export PATH=/usr/src/tensorrt/bin:$PATH
   ```

## Use the right environment

Do not trust only the repo dev environment for runtime work.

For real Jetson AGX Thor runtime checks, prefer a dedicated venv:

```bash
uv venv \
  --no-project \
  --no-managed-python \
  --python /usr/bin/python3 \
  --system-site-packages \
  .venv-thor
```

Then install:

```bash
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

## Known pitfalls

- `lerobot` must be installed with `--no-deps`.
- `tool.uv.sources.openpi = { workspace = true }` is required in the host root.
- old `ml-dtypes==0.4.1` pins must be replaced, not duplicated.
- older upstream `openpi` may also need source patches in:
  - `src/openpi/training/data_loader.py`
  - `src/openpi/transforms.py`
- the `transforms.py` patch is not just logic replacement; it may also need `Any` added to the
  `typing` import.
- a backup checkout with the same package name under `packages/` breaks `uv` resolution.
- `prepare-engine` imports `calibration.py`, so `torchvision` must be available even for FP16.
- `openpi_on_thor` is tutorial/reference only; runtime code should not depend on it.
- TensorRT builds can take many minutes; a zero-byte engine file during build is normal.

## FP8 / NVFP4 lessons already learned

- Keep `32` as the default fp8 calibration count unless new evidence says otherwise. The tested
  Thor stack did not show a plain-fp8 accuracy gain from raising it to `128`.
- The public `--enable-llm-nvfp4` path is attention-only:
  - all Gemma attention layers use NVFP4
  - explicit attention-matmul quantization is enabled
  - Gemma MLP stays on fp8
- Do not revert the public path to the earlier Gemma MLP weight-only fallback. That path kept
  accuracy closer than broad full-layer NVFP4, but it was slower than plain fp8 because TensorRT
  lowered it into cast/dequant-heavy kernels.
- Do not promote broad attention+MLP full-layer NVFP4 casually. It regressed badly after
  ONNX/TensorRT lowering even when quantized PyTorch still looked reasonable.
- For NVFP4 work, always inspect `trtexec` layer profiles. Good candidates avoid
  `ReplCastMulCast`-style dominance and preserve fused attention kernels.

## Debug order for backend quality

If outputs are wrong, use:

1. JAX -> PyTorch conversion
2. PyTorch vs JAX validation
3. FP16 ONNX export
4. strongly typed FP16 TensorRT build
5. TensorRT vs JAX validation
6. only then FP8/NVFP4

For Gemma-based `pi05_*`, keep strongly typed TensorRT on by default.

For deeper NVFP4 investigation, keep this extra order in mind:

1. compare candidate TensorRT against plain fp8 TensorRT
2. compare the same candidate against JAX on the same fixed real-data examples
3. inspect `trtexec` profiles before trusting a speedup claim
4. only then decide whether a scoped NVFP4 candidate is worth promoting

## Useful commands

Doctor:

```bash
.venv-thor/bin/openpi-thor doctor
```

Status:

```bash
.venv-thor/bin/openpi-thor status --bundle-dir /path/to/bundle --verbose
```

Real pipeline:

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

## When editing code

- Add or update a regression test whenever you fix a companion-workflow issue.
- If you change host integration, test both:
  - fixture-based patching behavior
  - the real host checkout shape
- If you change bundle metadata, check:
  - `openpi_thor_bundle.json`
  - `reports/*.json`
