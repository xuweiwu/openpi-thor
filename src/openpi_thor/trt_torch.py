from __future__ import annotations

import atexit
import ctypes
import os
from collections.abc import Iterable

import torch


def _import_trt():
    import tensorrt as trt

    return trt


def torch_type(trt_type) -> torch.dtype:
    trt = _import_trt()
    mapping = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int8: torch.int8,
        trt.int32: torch.int32,
        trt.bool: torch.bool,
        trt.uint8: torch.uint8,
        trt.int64: torch.int64,
    }
    if trt_type in mapping:
        return mapping[trt_type]

    raise TypeError(f"Could not resolve TensorRT datatype to an equivalent torch dtype: {trt_type}")


class Engine:
    """Small TensorRT engine wrapper re-homed from the Jetson tutorial helpers."""

    def __init__(self, file: str, plugins: Iterable[str] = ()) -> None:
        trt = _import_trt()
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, "")

        self.plugins = [ctypes.CDLL(plugin, ctypes.RTLD_GLOBAL) for plugin in plugins]
        self.file = file
        self.load(file)

        def destroy(engine: "Engine") -> None:
            del engine.execution_context
            del engine.handle

        atexit.register(destroy, self)
        self.print()

    def print(self) -> None:
        if int(os.getenv("LOCAL_RANK", -1)) not in [0, -1]:
            return

        print("============= TRT Engine Detail =============")
        print(f"Engine file: {self.file}")
        print(f"Inputs: {len(self.in_meta)}")
        for index, item in enumerate(self.in_meta):
            tensor_name, shape, dtype = item[:3]
            print(f"   {index}. {tensor_name}: {'x'.join(map(str, shape))} [{dtype}]")

        print(f"Outputs: {len(self.out_meta)}")
        for index, item in enumerate(self.out_meta):
            tensor_name, shape, dtype = item[:3]
            print(f"   {index}. {tensor_name}: {'x'.join(map(str, shape))} [{dtype}]")
        print("=============================================")

    def load(self, file: str) -> None:
        trt = _import_trt()
        runtime = trt.Runtime(self.logger)

        with open(file, "rb") as handle:
            self.handle = runtime.deserialize_cuda_engine(handle.read())
            assert self.handle is not None, f"Failed to deserialize the cuda engine from file: {file}"

        self.execution_context = self.handle.create_execution_context()
        self.meta = []
        self.in_meta = []
        self.out_meta = []
        for tensor_name in self.handle:
            shape = self.handle.get_tensor_shape(tensor_name)
            dtype = torch_type(self.handle.get_tensor_dtype(tensor_name))
            if self.handle.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.in_meta.append([tensor_name, shape, dtype])
            else:
                self.out_meta.append([tensor_name, shape, dtype])

    def __call__(self, *args, **inputs):
        return self.forward(*args, **inputs)

    def set_runtime_tensor_shape(self, name, shape) -> None:
        self.execution_context.set_input_shape(name, shape)

    def forward(self, *args, **kwargs):
        return_list = kwargs.pop("return_list", False)
        reference_tensors = []
        stream = torch.cuda.current_stream()
        for index, x in enumerate(args):
            name, _shape, dtype = self.in_meta[index]
            runtime_shape = self.execution_context.get_tensor_shape(name)
            assert isinstance(x, torch.Tensor), f"Unsupported tensor type: {type(x)}"
            assert runtime_shape == x.shape, f"Invalid input shape: {runtime_shape} != {x.shape}"
            assert dtype == x.dtype, f"Invalid tensor dtype, expected dtype is {dtype}, but got {x.dtype}"
            assert x.is_cuda, f"Invalid tensor device, expected device is cuda, but got {x.device}"
            x = x.cuda().contiguous()
            self.execution_context.set_tensor_address(name, x.data_ptr())
            reference_tensors.append(x)

        for name, _shape, dtype in self.in_meta:
            if name not in kwargs:
                continue

            runtime_shape = self.execution_context.get_tensor_shape(name)
            x = kwargs[name]
            assert isinstance(x, torch.Tensor), f"Unsupported tensor[{name}] type: {type(x)}"
            assert runtime_shape == x.shape, (
                f"Invalid input[{name}] shape: {x.shape}, but the expected shape is: {runtime_shape}"
            )
            assert dtype == x.dtype, f"Invalid tensor[{name}] dtype, expected dtype is {dtype}, but got {x.dtype}"
            assert x.is_cuda, f"Invalid tensor[{name}] device, expected device is cuda, but got {x.device}"
            x = x.cuda().contiguous()
            self.execution_context.set_tensor_address(name, x.data_ptr())
            reference_tensors.append(x)

        for name, _shape, dtype in self.out_meta:
            runtime_shape = self.execution_context.get_tensor_shape(name)
            output_tensor = torch.zeros(*runtime_shape, dtype=dtype, device=reference_tensors[0].device)
            self.execution_context.set_tensor_address(name, output_tensor.data_ptr())
            reference_tensors.append(output_tensor)

        self.execution_context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        assert len(reference_tensors) == len(self.in_meta) + len(self.out_meta), (
            "Invalid input tensors. "
            f"The expected I/O tensors are {len(self.in_meta) + len(self.out_meta)}, "
            f"but got {len(reference_tensors)}"
        )

        if return_list:
            return [reference_tensors[len(self.in_meta) + index] for index, _item in enumerate(self.out_meta)]
        return {item[0]: reference_tensors[len(self.in_meta) + index] for index, item in enumerate(self.out_meta)}
