import inspect

from openpi_thor import calibration
from openpi_thor import cli
from openpi_thor import convert
from openpi_thor import doctor
from openpi_thor import engine
from openpi_thor import export
from openpi_thor import host_integration
from openpi_thor import runtime
from openpi_thor import server
from openpi_thor import validate
from openpi_thor import workflow
from openpi_thor._schema import ArtifactBundle
from openpi_thor._schema import ArtifactRecord
from openpi_thor._schema import CheckpointLoadReport
from openpi_thor._schema import DoctorReport
from openpi_thor._schema import EngineProfile
from openpi_thor._schema import ExportOptions
from openpi_thor._schema import ValidationReport


def test_representative_public_symbols_have_docstrings() -> None:
    symbols = [
        cli.DoctorCommand,
        cli.ConvertJaxCommand,
        cli.ExportOnnxCommand,
        cli.BuildEngineCommand,
        cli.ValidateCommand,
        cli.ValidateTensorRTCommand,
        cli.PrepareEngineCommand,
        cli.StatusCommand,
        cli.ServeCommand,
        cli.main,
        convert.inspect_jax_checkpoint,
        convert.convert_jax_checkpoint,
        engine.build_engine,
        export.QuantizedMatMul,
        export.ONNXWrapper,
        export.export_to_onnx_bundle,
        workflow.prepare_engine,
        workflow.bundle_status,
        validate.compare_backends,
        validate.compare_tensorrt_engines,
        runtime.load_pytorch_bundle,
        runtime.load_tensorrt_policy,
        calibration.sample_dataset_examples,
        calibration.CalibrationBatches,
        calibration.IterableCalibrationSource,
        calibration.LeRobotPi05CalibrationSource,
        calibration.DummyCalibrationSource,
        calibration.build_calibration_batches,
        host_integration.HostPatchPlan,
        host_integration.plan_host_pyproject_patch,
        host_integration.write_host_pyproject_patch,
        host_integration.doctor_host_integration_warnings,
        doctor.run_doctor,
        server.serve,
        ExportOptions,
        EngineProfile,
        CheckpointLoadReport,
        ValidationReport,
        DoctorReport,
        ArtifactRecord,
        ArtifactBundle,
    ]

    for symbol in symbols:
        assert inspect.getdoc(symbol), f"missing docstring for {symbol!r}"
