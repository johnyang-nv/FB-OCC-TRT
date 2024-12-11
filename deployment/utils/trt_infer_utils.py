# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.
import os
import argparse
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import ctypes
import time
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from tqdm import tqdm

import sys
sys.path.append(".")

from mmdet3d.models.builder import build_model
from mmdet3d.datasets.builder import build_dataloader, build_dataset

BATCH_SIZE = 1

FP='fp32'
PLUGIN_LIBRARY1 = "/FB-BEV/TensorRT/lib/libtensorrt_ops.so"
ENGINE_PATH = "/FB-BEV/data/onnx/fbocc-r50-cbgs_depth_16f_16x4_20e_trt_fp32.engine"
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
trt.init_libnvinfer_plugins(TRT_LOGGER, '')
ctypes.cdll.LoadLibrary(PLUGIN_LIBRARY1)
PLUGIN_CREATORS = trt.get_plugin_registry().plugin_creator_list


class OutputAllocator(trt.IOutputAllocator):
    def __init__(self, curr_size):
        trt.IOutputAllocator.__init__(self)
        self.curr_size = curr_size
        self.allocated_mem = None
        if curr_size > 0:
            self.allocated_mem = cuda.mem_alloc(curr_size)
        self.tensor_shape = None

    def reallocate_output(self, tensor_name, memory, size, alignment):
        assert size > 0
        if size > self.curr_size:
            self.allocated_mem = cuda.mem_alloc(size)
        return int(self.allocated_mem)

    def notify_shape(self, tensor_name, shape):
        self.tensor_shape = shape

class HostDeviceMem(object):
    def __init__(self, name, host_mem, device_mem):
        """Within this context, host_mom means the cpu memory and device means the GPU memory
        """
        self.name = name
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return (
            "Name:\n"
            + str(self.name)
            + "\nHost:\n"
            + str(self.host)
            + "\nDevice:\n"
            + str(self.device)
            + "\n"
        )

    def __repr__(self):
        return self.__str__()
    
def build_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        loaded_engine = runtime.deserialize_cuda_engine(f.read())
    return loaded_engine

# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine, input_shapes, output_shapes):
    inputs = []
    outputs = []
    bindings = []
    
    stream = cuda.Stream()
    context = engine.create_execution_context()
    output_allocators = {}
    for binding_id, binding in enumerate(engine):
        if engine.binding_is_input(binding):
            dims = input_shapes[binding]
            # set binding shape for dynamic inputs
            context.set_binding_shape(binding_id, input_shapes[binding])
        else:
            output_allocator = OutputAllocator(0)
            context.set_output_allocator(binding, output_allocator)
            output_allocators[binding] = output_allocator
            dims = output_shapes[binding]
        size = trt.volume(dims) 
        if size==0:
            size=1
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        try:
            host_mem = cuda.pagelocked_empty(size, dtype)
        except:
            import pdb; pdb.set_trace()
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(binding, host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(binding, host_mem, device_mem))
    return inputs, outputs, bindings, stream,  output_allocators, context

def do_inference(context, bindings, inputs, outputs, stream, engine, batch_size=1, output_allocators=None):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    t1 = time.time()
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle) 
    stream.synchronize()
    t2 = time.time()
    out_mem={}
    assert output_allocators is not None
    for output in output_allocators:
        try:
            assert output_allocators[output].allocated_mem
        except:
            print('empty outputs detected')
        shape = output_allocators[output].tensor_shape
        try:
            assert shape is not None
        except:
            import pdb; pdb.set_trace()
        size = trt.volume(shape)
        dtype = trt.nptype(engine.get_tensor_dtype(output))
        output_buffer = cuda.pagelocked_empty(size, dtype)
        output_memory = context.get_tensor_address(output)
        output_mem = (output_buffer, output_memory)
        # Store tensor to output buffer and output memory mappings.
        out_mem[output] = output_mem
        cuda.memcpy_dtoh_async(output_mem[0], output_mem[1], stream)
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    results = {}
    for output in out_mem:
        output_mem = out_mem[output][0]
        # Get real output tensor size
        shape = output_allocators[output].tensor_shape
        # notify_shape must be called by TensorRT
        assert shape is not None
        size = trt.volume(shape)
        output_mem = output_mem[:size]
        output_mem = output_mem.reshape(shape)
        results[output] = output_mem

    return results, t2-t1


def infer(engine, trt_inputs, batch_size, input_shapes, output_shapes):
    inputs, outputs, bindings, stream,  output_allocators, context = allocate_buffers(engine, input_shapes, output_shapes)
    for id, it in enumerate(trt_inputs):
        
        if 0 not in it.shape:
            try:
                np.copyto(inputs[id].host, it.ravel().astype(inputs[id].host.dtype))
            except:
                print(id)
                import pdb; pdb.set_trace()
    outputs_h,t = do_inference(context, bindings, inputs, outputs, stream, engine, batch_size, output_allocators)
    return outputs_h,t

def run_trt(trt_inputs, engine, batch_size=1, input_shapes=None, output_shapes=None):
    outputs, t = infer(engine, trt_inputs, batch_size, input_shapes, output_shapes)
    return outputs, t





def build_engine_onnx(model_file):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:

        with open(model_file, 'rb') as model:
            parser.parse(model.read())

        network.mark_output(network.get_layer(network.num_layers - 1).get_output(0))

        return builder.build_cuda_engine(network)

