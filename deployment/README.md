# FB-OCC TensorRT Deployment on NVIDIA Drive Platform


This section provides a deployment workflow for **FB-OCC** using **TensorRT**, supporting both `FP32` and `FP16` inference. It includes all necessary components to streamline the process from model export to inference on **NVIDIA DRIVE Orin** with TensorRT.

<div align="center">

|                           | mIoU              | Latency (ms) on A40  |  Latency (ms) on NVIDIA DRIVE Orin   |
|---------------------------|-------------------|----------------------|--------------------------------------|
| FB-OCC-TensorRT_fp32      | 38.90             | 54.37                | 197.89                               |
| FB-OCC-TensorRT_fp16      | 38.86             | 34.26                | 138.62                               |
| FB-OCC-PyTorch (original) | 38.90 (reproduced)| 98.37                | -                                    |

</div>


The FB-OCC model achieves consistent accuracy across `FP32` and `FP16` precision levels. 
TensorRT models demonstrate lower latency compared to the original PyTorch implementation, with `FP16` further reducing inference time. 



## Prerequisites

   FB-OCC uses operations that are not natively supported by TensorRT, including `GridSample3D`, `BevPoolv2`, and `Multi-Scale Deformable Attention`. To ensure FB-OCC works correctly during inference, these operations need to be implemented as custom TensorRT plugins and compiled beforehand. These plugins expand TensorRT's functionality, allowing the model to operate as expected.
   
   You can modify the TensorRT plugins provided in the [BEVFormer_tensorrt](https://github.com/DerryHub/BEVFormer_tensorrt) repository to make them compatible with FB-OCC:
   ```bash
   # Clone the BEVFormer_tensorrt repository
   git clone https://github.com/DerryHub/BEVFormer_tensorrt
   # Checkout the specific commit for compatibility
   git checkout 303d314
   ```

## ONNX Export  
   First, refer to the [FB-BEV Repository Installation Guide](docs/install.md) for detailed installation instructions for the FB-OCC repository.

1. **Adapting TensorRT Functions for FB-OCC**
   
   The `trt_functions` from the [BEVFormer_tensorrt plugin](https://github.com/DerryHub/BEVFormer_tensorrt/tree/303d3140c14016047c07f9db73312af364f0dd7c/det2trt/models/functions) must be copied into the FB-OCC workspace and adjusted for compatibility. Follow these steps:

   ```bash
   # Copy BEVFormer_tensorrt functions to the FB-OCC workspace
   cp /path/to/BEVFormer_tensorrt/det2trt/models/functions/*.py /path/to/FB-BEV/deployment/trt_functions/

   # Navigate to the target directory and apply the patch for FB-OCC
   cd deployment/trt_functions/
   git apply FB-OCC_fn-patch-on-derryhub_fn.patch
   ```

2. **Generating the ONNX File**

   Run the following command to create the ONNX file for FB-OCC:
   ```bash
   python deployment/pth2onnx.py occupancy_configs/fb_occ/fbocc-r50-cbgs_depth_16f_16x4_20e_trt.py
   ```

   This command uses real data samples and saves the input data for use in later steps when creating the TensorRT engine.


   

## TensorRT Plugin Cross-Compilation for DRIVE OS Linux on x86 host

   Cross-compiling is essential when the target platform, such as NVIDIA DRIVE OS Linux, differs from the development environment (x86 host). It allows developers to build ARM-compatible plugins on the x86 host by leveraging its computational power. 

1. **Apply the Patch**

   To prepare the TensorRT plugins for cross-compilation, apply the `FB-OCC_trt_plugin_aarch64.patch` to the plugin files:

   ```bash
   cd /path/to/BEVFormer_tensorrt/
   git apply /path/to/FB-BEV/deployment/plugins/FB-OCC_trt_plugin_aarch64.patch
   ```   

2. **Set Up the Environment**

   Download a pre-configured NGC container to streamline the cross-compilation process. Detailed instructions for accessing NGC containers are available in the [NGC Container Setup Guide](https://developer.nvidia.com/docs/drive/drive-os/6.0.10/public/drive-os-linux-installation/common/topics/installation/docker-ngc/setup-drive-os-linux-nvonline.html).    
   - Download the `nv-tensorrt-repo-ubuntu2004-cuda11.4-trt8.6.13.3-d6l-cross-ga-20240202_1-1_amd64.deb` debian package to your workspace `/path/to/BEVFormer_tensorrt/` from [Poduct Information Delivery](https://apps.nvidia.com/PID/ContentGroup/Detail/1948?FromLocation=CL) with your NVONLINE account. 
   - Launch the Drive OS Linux Docker container with `BEVFormer_tensorrt` mounted::
   ```bash
   docker run --gpus all -it --network=host --rm \
     -v /your/path/to/BEVFormer_tensorrt/:/BEVFormer_tensorrt \
     nvcr.io/drive/driveos-sdk/drive-agx-orin-linux-aarch64-sdk-build-x86:6.0.10.0-0009
   ```
3. **Install Required Components**

   Inside the Docker container, execute the following commands to install the necessary components and build the plugins:   
   ```bash
   dpkg -i /BEVFormer_tensorrt/nv-tensorrt-repo-ubuntu2004-cuda11.4-trt8.6.13.3-d6l-cross-ga-20240202_1-1_amd64.deb
   apt install tensorrt-safe-cross-aarch64
   cd /BEVFormer_tensorrt/TensorRT/
   make TARGET=aarch64
   ```
   When completed, the compiled plugin file will be located at `/drive/bin/aarch64/FB-OCC_trt_plugin_aarch64.so`.

   `FB-OCC_trt_plugin_aarch_aarch64.so` will be used in the next steps to create the TensorRT engine.

   
## Running TensorRT Engine Creation on the Target Platform

TensorRT engine creation must be performed on the target platform running NVIDIA DRIVE OS, as the cross-compiled TensorRT plugin is not recognized on x86 systems but works on ARM-based platforms.

1. **Transfer the Environment and Plugin**
   
   Transfer the workspace and the cross-compiled plugin (e.g., `FB-OCC_trt_plugin_aarch64.so`) to the target platform. Ensure real data samples are provided to satisfy the model's dynamic input size requirements.

   **Important**: Real data samples are required during engine creation to avoid errors. These samples must align with the model’s dynamic input size requirements.

2. **Run the Engine Creation Command** 
   
   Navigate to the FB-BEV directory on the target platform and execute the following commands to create the TensorRT engine:

   ```bash
   cd /path/to/FB-BEV/
   # Standard engine creation
   python deployment/create_engine.py --trt_plugin_path /path/to/FB-OCC_trt_plugin_aarch64.so

   # Engine creation with FP16 precision
   python deployment/create_engine.py --trt_plugin_path /path/to/FB-OCC_trt_plugin_aarch64.so --fp16

   # Engine creation with a custom engine path
   python deployment/create_engine.py --trt_plugin_path /path/to/FB-OCC_trt_plugin_aarch64.so --trt_engine_path <path_to_TensorRT_engine>
   ```

3. **Output Location**

   Upon successful execution, the TensorRT engine will be saved at the specified `<path_to_TensorRT_engine>` or, by default, at `data/fbocc-r50-cbgs_depth_16f_16x4_20e_trt.engine`.

   #### **Notes:**

   - **Real Data Requirement:** Ensure real data samples are available and properly configured (e.g., .dat files) to avoid errors during engine creation. The model uses dynamic input sizes for multiple inputs.
   - **Dataset Configuration:** Verify that the dataset paths for the input files are correctly set up to ensure smooth engine creation.

## Validation / Inference 

   To validate the accuracy of the generated TensorRT engine, use the following command:

   ```bash
   python tools/test.py occupancy_configs/fb_occ/fbocc-r50-cbgs_depth_16f_16x4_20e_trt.py ckpts/fbocc-r50-cbgs_depth_16f_16x4_20e.pth --trt_engine <path_to_TensorRT_engine>
   ```
   
   Replace `<path_to_TensorRT_engine>` with the appropriate path to your TensorRT engine file. For example, `data/onnx/fbocc-r50-cbgs_depth_16f_16x4_20e_trt.engine`. 

   **Results**

   The TensorRT engine produces results consistent with the PyTorch implementation:
   ```bash
   FP32 Precision: 38.90
   FP16 Precision: 38.86
   ```
   These results align closely with the reproduced PyTorch accuracy of `38.90` from the original implementation.
   TensorRT models demonstrate reduced inference latency compared to the PyTorch implementation, with further improvements observed when using FP16 precision.

