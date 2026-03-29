from __future__ import annotations

import dataclasses
import json
from typing import Annotated

import tyro

from openpi_thor._schema import EngineProfile
from openpi_thor._schema import ExportOptions


@dataclasses.dataclass(frozen=True)
class DoctorCommand:
    pass


@dataclasses.dataclass(frozen=True)
class ConvertJaxCommand:
    config: str
    checkpoint_dir: str
    output_dir: str
    precision: str = "bfloat16"
    copy_assets: bool = True
    overwrite: bool = False


@dataclasses.dataclass(frozen=True)
class ExportOnnxCommand:
    config: str
    bundle_dir: str
    precision: str = "fp16"
    num_steps: int = 10
    enable_llm_nvfp4: bool = False
    quantize_attention_matmul: bool = False
    num_calibration_samples: int = 32
    allow_dummy_calibration: bool = False
    use_lerobot_calibration: bool = False
    dataset_repo_id: str | None = None
    dataset_root: str | None = None


@dataclasses.dataclass(frozen=True)
class BuildEngineCommand:
    config: str
    bundle_dir: str
    onnx_path: str | None = None
    min_batch: int = 1
    opt_batch: int = 1
    max_batch: int = 1
    min_seq_len: int | None = None
    opt_seq_len: int | None = None
    max_seq_len: int | None = None
    strongly_typed: bool = True
    dry_run: bool = False


@dataclasses.dataclass(frozen=True)
class ValidateCommand:
    config: str
    bundle_dir: str
    candidate_backend: str = "pytorch"
    reference_checkpoint_dir: str | None = None
    engine_path: str | None = None
    num_examples: int = 8
    dataset_repo_id: str | None = None
    dataset_root: str | None = None


@dataclasses.dataclass(frozen=True)
class PrepareEngineCommand:
    config: str
    bundle_dir: str
    precision: str = "fp16"
    num_steps: int = 10
    enable_llm_nvfp4: bool = False
    quantize_attention_matmul: bool = False
    num_calibration_samples: int = 32
    allow_dummy_calibration: bool = False
    validate: bool = False
    reference_checkpoint_dir: str | None = None
    validation_num_examples: int = 8
    dataset_repo_id: str | None = None
    dataset_root: str | None = None
    min_batch: int = 1
    opt_batch: int = 1
    max_batch: int = 1
    min_seq_len: int | None = None
    opt_seq_len: int | None = None
    max_seq_len: int | None = None
    strongly_typed: bool = True


@dataclasses.dataclass(frozen=True)
class StatusCommand:
    bundle_dir: str
    verbose: bool = False


@dataclasses.dataclass(frozen=True)
class ServeCommand:
    config: str
    bundle_dir: str
    engine_path: str | None = None
    host: str = "0.0.0.0"
    port: int = 8000
    require_validated: bool = True


Command = (
    Annotated[DoctorCommand, tyro.conf.subcommand(name="doctor")]
    | Annotated[ConvertJaxCommand, tyro.conf.subcommand(name="convert-jax")]
    | Annotated[ExportOnnxCommand, tyro.conf.subcommand(name="export-onnx")]
    | Annotated[BuildEngineCommand, tyro.conf.subcommand(name="build-engine")]
    | Annotated[ValidateCommand, tyro.conf.subcommand(name="validate")]
    | Annotated[PrepareEngineCommand, tyro.conf.subcommand(name="prepare-engine")]
    | Annotated[StatusCommand, tyro.conf.subcommand(name="status")]
    | Annotated[ServeCommand, tyro.conf.subcommand(name="serve")]
)


def main() -> None:
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
                command.output_dir,
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
