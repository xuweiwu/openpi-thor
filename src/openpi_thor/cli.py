from __future__ import annotations

import dataclasses
import json
from typing import Annotated

import tyro

from openpi_thor._schema import EngineProfile
from openpi_thor._schema import ExportOptions


@dataclasses.dataclass(frozen=True)
class DoctorCommand:
    """Inspect the Jetson AGX Thor runtime and host companion integration."""

    pass


@dataclasses.dataclass(frozen=True)
class ConvertJaxCommand:
    """Convert a JAX checkpoint into a reusable PyTorch bundle directory."""

    config: Annotated[
        str,
        tyro.conf.arg(help="Registered OpenPI training config name from the host repo, usually a pi05_* config."),
    ]
    checkpoint_dir: Annotated[
        str,
        tyro.conf.arg(help="Directory that contains the JAX checkpoint, typically the training output with params/."),
    ]
    bundle_dir: Annotated[
        str,
        tyro.conf.arg(help="Bundle directory to create or update with converted weights and copied assets."),
    ]
    precision: Annotated[
        str,
        tyro.conf.arg(help="PyTorch weight dtype to write into model.safetensors. Supported: float32 or bfloat16."),
    ] = "bfloat16"
    copy_assets: Annotated[
        bool,
        tyro.conf.arg(help="Copy the checkpoint assets/ directory into the bundle when one is available."),
    ] = True
    overwrite: Annotated[
        bool,
        tyro.conf.arg(help="Allow writing into a non-empty bundle directory instead of refusing the conversion."),
    ] = False


@dataclasses.dataclass(frozen=True)
class ExportOnnxCommand:
    """Export an ONNX graph from an existing PyTorch bundle."""

    config: Annotated[
        str,
        tyro.conf.arg(help="Registered OpenPI training config name from the host repo."),
    ]
    bundle_dir: Annotated[
        str,
        tyro.conf.arg(help="Existing bundle directory that contains model.safetensors and receives ONNX artifacts."),
    ]
    precision: Annotated[
        str,
        tyro.conf.arg(help="ONNX export precision. Start with fp16; use fp8 only when calibration data is available."),
    ] = "fp16"
    num_steps: Annotated[
        int,
        tyro.conf.arg(help="Number of denoising steps baked into the exported sampler graph."),
    ] = 10
    enable_llm_nvfp4: Annotated[
        bool,
        tyro.conf.arg(
            help=(
                "Enable the current openpi-thor NVFP4 path: Gemma MLP weights use NVFP4 while the rest of the "
                "language-model activations stay on FP8."
            )
        ),
    ] = False
    quantize_attention_matmul: Annotated[
        bool,
        tyro.conf.arg(help="Insert quantized attention matmul nodes so fp8/NVFP4 attention ops are explicit in ONNX."),
    ] = False
    num_calibration_samples: Annotated[
        int,
        tyro.conf.arg(help="Number of real samples to use for fp8 calibration when exporting quantized models."),
    ] = 32
    allow_dummy_calibration: Annotated[
        bool,
        tyro.conf.arg(help="Allow synthetic calibration instead of real dataset samples. Intended for debugging only."),
    ] = False
    use_lerobot_calibration: Annotated[
        bool,
        tyro.conf.arg(
            help=(
                "Compatibility no-op. Real fp8 export already defaults to LeRobot calibration unless dummy "
                "calibration is explicitly allowed."
            )
        ),
    ] = False
    dataset_repo_id: Annotated[
        str | None,
        tyro.conf.arg(help="Override the dataset repo id used for real calibration samples."),
    ] = None
    dataset_root: Annotated[
        str | None,
        tyro.conf.arg(help="Optional local LeRobot dataset root used together with --dataset-repo-id."),
    ] = None


@dataclasses.dataclass(frozen=True)
class BuildEngineCommand:
    """Build a TensorRT engine from an ONNX artifact recorded in the bundle."""

    config: Annotated[
        str,
        tyro.conf.arg(help="Registered OpenPI training config name from the host repo."),
    ]
    bundle_dir: Annotated[
        str,
        tyro.conf.arg(help="Bundle directory that contains the ONNX artifact and receives the built engine."),
    ]
    onnx_path: Annotated[
        str | None,
        tyro.conf.arg(help="Optional explicit ONNX path to build from. Defaults to the active ONNX artifact in the bundle."),
    ] = None
    min_batch: Annotated[
        int,
        tyro.conf.arg(help="Minimum batch size for the TensorRT optimization profile."),
    ] = 1
    opt_batch: Annotated[
        int,
        tyro.conf.arg(help="Preferred batch size for the TensorRT optimization profile."),
    ] = 1
    max_batch: Annotated[
        int,
        tyro.conf.arg(help="Maximum batch size for the TensorRT optimization profile."),
    ] = 1
    min_seq_len: Annotated[
        int | None,
        tyro.conf.arg(help="Optional minimum prompt sequence length override for the optimization profile."),
    ] = None
    opt_seq_len: Annotated[
        int | None,
        tyro.conf.arg(help="Optional preferred prompt sequence length override for the optimization profile."),
    ] = None
    max_seq_len: Annotated[
        int | None,
        tyro.conf.arg(help="Optional maximum prompt sequence length override for the optimization profile."),
    ] = None
    strongly_typed: Annotated[
        bool,
        tyro.conf.arg(
            help="Preserve explicit dtypes from the ONNX graph. Recommended on Jetson AGX Thor and enabled by default."
        ),
    ] = True
    dry_run: Annotated[
        bool,
        tyro.conf.arg(help="Print and record the trtexec command without actually building the TensorRT engine."),
    ] = False


@dataclasses.dataclass(frozen=True)
class ValidateCommand:
    """Compare PyTorch or TensorRT outputs against the original JAX checkpoint."""

    config: Annotated[
        str,
        tyro.conf.arg(help="Registered OpenPI training config name from the host repo."),
    ]
    bundle_dir: Annotated[
        str,
        tyro.conf.arg(help="Bundle directory that contains the converted weights, engines, and validation reports."),
    ]
    candidate_backend: Annotated[
        str,
        tyro.conf.arg(help="Backend to evaluate against JAX. Supported values are pytorch and tensorrt."),
    ] = "pytorch"
    reference_checkpoint_dir: Annotated[
        str | None,
        tyro.conf.arg(help="Optional JAX checkpoint directory. Defaults to the source checkpoint recorded in the bundle."),
    ] = None
    engine_path: Annotated[
        str | None,
        tyro.conf.arg(help="Specific TensorRT engine to validate when candidate-backend is tensorrt."),
    ] = None
    num_examples: Annotated[
        int,
        tyro.conf.arg(help="Number of dataset examples to compare with fixed-noise inference."),
    ] = 8
    dataset_repo_id: Annotated[
        str | None,
        tyro.conf.arg(help="Override the dataset repo id used to sample validation examples."),
    ] = None
    dataset_root: Annotated[
        str | None,
        tyro.conf.arg(help="Optional local LeRobot dataset root used together with --dataset-repo-id."),
    ] = None


@dataclasses.dataclass(frozen=True)
class ValidateTensorRTCommand:
    """Compare one TensorRT engine against another on the same dataset examples."""

    config: Annotated[
        str,
        tyro.conf.arg(help="Registered OpenPI training config name from the host repo."),
    ]
    bundle_dir: Annotated[
        str,
        tyro.conf.arg(help="Bundle directory used to resolve the recommended reference engine and write reports."),
    ]
    candidate_engine_path: Annotated[
        str,
        tyro.conf.arg(help="TensorRT engine path to evaluate as the candidate."),
    ]
    reference_engine_path: Annotated[
        str | None,
        tyro.conf.arg(help="Optional TensorRT engine path to use as the reference. Defaults to the bundle's recommended engine."),
    ] = None
    num_examples: Annotated[
        int,
        tyro.conf.arg(help="Number of dataset examples to compare with fixed-noise inference."),
    ] = 8
    dataset_repo_id: Annotated[
        str | None,
        tyro.conf.arg(help="Override the dataset repo id used to sample validation examples."),
    ] = None
    dataset_root: Annotated[
        str | None,
        tyro.conf.arg(help="Optional local LeRobot dataset root used together with --dataset-repo-id."),
    ] = None


@dataclasses.dataclass(frozen=True)
class PrepareEngineCommand:
    """Export ONNX, build TensorRT, and optionally validate in one command."""

    config: Annotated[
        str,
        tyro.conf.arg(help="Registered OpenPI training config name from the host repo."),
    ]
    bundle_dir: Annotated[
        str,
        tyro.conf.arg(help="Bundle directory to read/update across export, build, and optional validation."),
    ]
    precision: Annotated[
        str,
        tyro.conf.arg(help="Export precision to prepare. Use fp16 first; use fp8 only when calibration data is ready."),
    ] = "fp16"
    num_steps: Annotated[
        int,
        tyro.conf.arg(help="Number of denoising steps baked into the exported sampler graph."),
    ] = 10
    enable_llm_nvfp4: Annotated[
        bool,
        tyro.conf.arg(
            help=(
                "Enable the current openpi-thor NVFP4 path: Gemma MLP weights use NVFP4 while the rest of the "
                "language-model activations stay on FP8."
            )
        ),
    ] = False
    quantize_attention_matmul: Annotated[
        bool,
        tyro.conf.arg(help="Insert quantized attention matmul nodes before exporting fp8 or NVFP4 graphs."),
    ] = False
    num_calibration_samples: Annotated[
        int,
        tyro.conf.arg(help="Number of real samples to use for fp8 calibration."),
    ] = 32
    allow_dummy_calibration: Annotated[
        bool,
        tyro.conf.arg(help="Allow synthetic calibration instead of real dataset samples. Intended for debugging only."),
    ] = False
    validate: Annotated[
        bool,
        tyro.conf.arg(help="Run a JAX-versus-TensorRT validation pass after building the engine."),
    ] = False
    reference_checkpoint_dir: Annotated[
        str | None,
        tyro.conf.arg(help="Optional JAX checkpoint directory to use for validation."),
    ] = None
    validation_num_examples: Annotated[
        int,
        tyro.conf.arg(help="Number of dataset examples to use if --validate is enabled."),
    ] = 8
    dataset_repo_id: Annotated[
        str | None,
        tyro.conf.arg(help="Override the dataset repo id used for calibration and validation."),
    ] = None
    dataset_root: Annotated[
        str | None,
        tyro.conf.arg(help="Optional local LeRobot dataset root used together with --dataset-repo-id."),
    ] = None
    min_batch: Annotated[
        int,
        tyro.conf.arg(help="Minimum batch size for the TensorRT optimization profile."),
    ] = 1
    opt_batch: Annotated[
        int,
        tyro.conf.arg(help="Preferred batch size for the TensorRT optimization profile."),
    ] = 1
    max_batch: Annotated[
        int,
        tyro.conf.arg(help="Maximum batch size for the TensorRT optimization profile."),
    ] = 1
    min_seq_len: Annotated[
        int | None,
        tyro.conf.arg(help="Optional minimum prompt sequence length override for the optimization profile."),
    ] = None
    opt_seq_len: Annotated[
        int | None,
        tyro.conf.arg(help="Optional preferred prompt sequence length override for the optimization profile."),
    ] = None
    max_seq_len: Annotated[
        int | None,
        tyro.conf.arg(help="Optional maximum prompt sequence length override for the optimization profile."),
    ] = None
    strongly_typed: Annotated[
        bool,
        tyro.conf.arg(
            help="Preserve explicit dtypes from the ONNX graph during engine build. Recommended and enabled by default."
        ),
    ] = True


@dataclasses.dataclass(frozen=True)
class StatusCommand:
    """Summarize bundle contents, report files, and the recommended engine."""

    bundle_dir: Annotated[
        str,
        tyro.conf.arg(help="Bundle directory to inspect."),
    ]
    verbose: Annotated[
        bool,
        tyro.conf.arg(help="Also load the JSON report payloads referenced by the bundle manifest."),
    ] = False


@dataclasses.dataclass(frozen=True)
class ServeCommand:
    """Start the websocket inference server for the selected or recommended engine."""

    config: Annotated[
        str,
        tyro.conf.arg(help="Registered OpenPI training config name from the host repo."),
    ]
    bundle_dir: Annotated[
        str,
        tyro.conf.arg(help="Bundle directory that contains the engine and metadata needed for serving."),
    ]
    engine_path: Annotated[
        str | None,
        tyro.conf.arg(help="Optional explicit engine path. Defaults to the bundle's recommended engine when present."),
    ] = None
    host: Annotated[
        str,
        tyro.conf.arg(help="Host interface to bind the websocket server to."),
    ] = "0.0.0.0"
    port: Annotated[
        int,
        tyro.conf.arg(help="TCP port for the websocket server."),
    ] = 8000
    require_validated: Annotated[
        bool,
        tyro.conf.arg(help="Refuse to serve unvalidated fp8/NVFP4 artifacts unless explicitly disabled."),
    ] = True


Command = (
    Annotated[
        DoctorCommand,
        tyro.conf.subcommand(name="doctor", description="Check the runtime, TensorRT tools, and host integration."),
    ]
    | Annotated[
        ConvertJaxCommand,
        tyro.conf.subcommand(name="convert-jax", description="Convert a JAX checkpoint into a PyTorch bundle."),
    ]
    | Annotated[
        ExportOnnxCommand,
        tyro.conf.subcommand(name="export-onnx", description="Export an ONNX graph from an existing bundle."),
    ]
    | Annotated[
        BuildEngineCommand,
        tyro.conf.subcommand(name="build-engine", description="Build a TensorRT engine from a bundle's ONNX artifact."),
    ]
    | Annotated[
        ValidateCommand,
        tyro.conf.subcommand(name="validate", description="Compare PyTorch or TensorRT outputs against the JAX reference."),
    ]
    | Annotated[
        ValidateTensorRTCommand,
        tyro.conf.subcommand(
            name="validate-tensorrt",
            description="Compare one TensorRT engine directly against another engine.",
        ),
    ]
    | Annotated[
        PrepareEngineCommand,
        tyro.conf.subcommand(name="prepare-engine", description="Export ONNX, build TensorRT, and optionally validate."),
    ]
    | Annotated[
        StatusCommand,
        tyro.conf.subcommand(name="status", description="Show bundle contents, reports, and the recommended engine."),
    ]
    | Annotated[
        ServeCommand,
        tyro.conf.subcommand(name="serve", description="Start the websocket inference server."),
    ]
)


def main() -> None:
    """Parse CLI arguments and dispatch to the selected openpi-thor command."""

    command = tyro.cli(Command)
    match command:
        case DoctorCommand():
            from openpi_thor.doctor import run_doctor

            print(json.dumps(run_doctor().to_dict(), indent=2))
        case ConvertJaxCommand():
            from openpi_thor.convert import convert_jax_checkpoint

            bundle = convert_jax_checkpoint(
                command.config,
                command.checkpoint_dir,
                command.bundle_dir,
                precision=command.precision,
                copy_assets=command.copy_assets,
                overwrite=command.overwrite,
            )
            print(bundle.metadata_path)
        case ExportOnnxCommand():
            from openpi_thor.export import export_to_onnx_bundle

            bundle = export_to_onnx_bundle(
                command.config,
                command.bundle_dir,
                options=ExportOptions(
                    precision=command.precision,
                    num_steps=command.num_steps,
                    enable_llm_nvfp4=command.enable_llm_nvfp4,
                    quantize_attention_matmul=command.quantize_attention_matmul,
                    num_calibration_samples=command.num_calibration_samples,
                    allow_dummy_calibration=command.allow_dummy_calibration,
                ),
                dataset_repo_id=command.dataset_repo_id,
                dataset_root=command.dataset_root,
            )
            print(bundle.metadata_path)
        case BuildEngineCommand():
            from openpi_thor.engine import build_engine

            bundle = build_engine(
                command.config,
                command.bundle_dir,
                onnx_path=command.onnx_path,
                profile=EngineProfile(
                    min_batch=command.min_batch,
                    opt_batch=command.opt_batch,
                    max_batch=command.max_batch,
                    min_seq_len=command.min_seq_len,
                    opt_seq_len=command.opt_seq_len,
                    max_seq_len=command.max_seq_len,
                    strongly_typed=command.strongly_typed,
                ),
                dry_run=command.dry_run,
            )
            print(bundle.metadata_path)
        case ValidateCommand():
            from openpi_thor.validate import compare_backends

            report = compare_backends(
                command.config,
                command.bundle_dir,
                candidate_backend=command.candidate_backend,
                reference_checkpoint_dir=command.reference_checkpoint_dir,
                engine_path=command.engine_path,
                num_examples=command.num_examples,
                dataset_repo_id=command.dataset_repo_id,
                dataset_root=command.dataset_root,
            )
            print(json.dumps(report.to_dict(), indent=2))
        case ValidateTensorRTCommand():
            from openpi_thor.validate import compare_tensorrt_engines

            report = compare_tensorrt_engines(
                command.config,
                command.bundle_dir,
                candidate_engine_path=command.candidate_engine_path,
                reference_engine_path=command.reference_engine_path,
                num_examples=command.num_examples,
                dataset_repo_id=command.dataset_repo_id,
                dataset_root=command.dataset_root,
            )
            print(json.dumps(report.to_dict(), indent=2))
        case PrepareEngineCommand():
            from openpi_thor.workflow import prepare_engine

            bundle = prepare_engine(
                command.config,
                command.bundle_dir,
                export_options=ExportOptions(
                    precision=command.precision,
                    num_steps=command.num_steps,
                    enable_llm_nvfp4=command.enable_llm_nvfp4,
                    quantize_attention_matmul=command.quantize_attention_matmul,
                    num_calibration_samples=command.num_calibration_samples,
                    allow_dummy_calibration=command.allow_dummy_calibration,
                ),
                profile=EngineProfile(
                    min_batch=command.min_batch,
                    opt_batch=command.opt_batch,
                    max_batch=command.max_batch,
                    min_seq_len=command.min_seq_len,
                    opt_seq_len=command.opt_seq_len,
                    max_seq_len=command.max_seq_len,
                    strongly_typed=command.strongly_typed,
                ),
                validate=command.validate,
                reference_checkpoint_dir=command.reference_checkpoint_dir,
                validation_num_examples=command.validation_num_examples,
                dataset_repo_id=command.dataset_repo_id,
                dataset_root=command.dataset_root,
            )
            print(bundle.metadata_path)
        case StatusCommand():
            from openpi_thor.workflow import bundle_status

            print(json.dumps(bundle_status(command.bundle_dir, verbose=command.verbose), indent=2))
        case ServeCommand():
            from openpi_thor.server import serve

            serve(
                command.config,
                command.bundle_dir,
                engine_path=command.engine_path,
                host=command.host,
                port=command.port,
                require_validated=command.require_validated,
            )


if __name__ == "__main__":
    main()
