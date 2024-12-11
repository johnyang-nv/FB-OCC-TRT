# FB-OCC TensorRT Deployment on NVIDIA Drive Platform


This repository provides a comprehensive deployment framework for **FB-OCC** on the **NVIDIA DRIVE platform** using **TensorRT**, supporting both `FP32` and `FP16` inference for optimized performance. It includes all necessary components to streamline the process from model export to efficient execution on TensorRT for NVIDIA DRIVE deployments.

<div align="center">

|                      | mIoU           | Latency (ms) on A40  |  Latency (ms) on NVIDIA DRIVE Orin   |
|---------------------------|----------------|--------------|--------|
| FB-OCC-TensorRT_fp32      | 38.90          | 54.37        | 197.89 |
| FB-OCC-TensorRT_fp16      | 38.86          | 34.26        | 138.62 |
| FB-OCC-PyTorch (original) | 38.90 (reproduced)| 98.37    | -      |

</div>


The FB-OCC model achieves consistent accuracy across `FP32` and `FP16` precision levels. 
TensorRT models demonstrate significantly lower latency compared to the original PyTorch implementation, with `FP16` further reducing inference time. 



1. **Prerequisites**
   
   Download the `TensorRT-8.6.13.3` tarball from the [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt). TensorRT is available for free as a binary on multiple platforms or as a container on NVIDIA NGC. After downloading, follow the installation instructions provided on the page to set up TensorRT on your system.

   ```bash
   tar -xzvf <path_to_your_TensorRT_tarball>
   export LD_LIBRARY_PATH=<path_to_your_TensorRT>/lib:$LD_LIBRARY_PATH
   ```

2. **TensorRT Plugin Compilation**  

   FB-OCC utilizes operations that are not natively supported in TensorRT, such as `GridSample3D`, `BevPoolv2` and `Multi-Scale Deformable Attention`. These operations must be implemented as custom TensorRT plugins and compiled before using FB-OCC for inference. These plugins extend TensorRT’s capabilities, enabling the model to function as expected during optimization and deployment.

   The TensorRT plugins provided in the [BEVFormer_tensorrt](https://github.com/DerryHub/BEVFormer_tensorrt) repository can be modified to work with FB-OCC by applying the necessary tweaks.

   ```bash
   git clone https://github.com/DerryHub/BEVFormer_tensorrt
   git checkout 303d314
   cd BEVFormer_tensorrt/
   ```

   ### Cross-compile the plugin for DRIVE OS Linux on x86 host

   Cross-compiling is essential when the target platform, such as NVIDIA DRIVE OS Linux, differs from the development environment (x86 host). It allows developers to build ARM-compatible plugins on the x86 host by leveraging its computational power. 
   
   To cross-compile the plugins, apply the `FB-OCC_trt_plugin_aarch64.patch` to the TensorRT plugins. This process reorganizes the plugin files to prepare them for subsequent steps.

   ```bash
   cd TensorRT/
   git apply /your/path/to/FB-BEV/deployment/plugins/FB-OCC_trt_plugin_aarch64.patch
   ```

   A pre-configured environment for subsequent steps can be conveniently obtained by downloading one of the NGC containers. Detailed instructions on accessing such a container are available at [NGC Container Setup Guide](https://developer.nvidia.com/docs/drive/drive-os/6.0.10/public/drive-os-linux-installation/common/topics/installation/docker-ngc/setup-drive-os-linux-nvonline.html).    
   Download the `nv-tensorrt-repo-ubuntu2004-cuda11.4-trt8.6.13.3-d6l-cross-ga-20240202_1-1_amd64.deb` debian package from [Poduct Information Delivery](https://apps.nvidia.com/PID/ContentGroup/Detail/1948?FromLocation=CL) with your NVONLINE account. 
   
   After downloading, proceed to launch the Drive OS Linux Docker image.

   ```bash
   docker run --gpus all -it --network=host --rm -v /your/path/to/BEVFormer_tensorrt/TensorRT:/workspace/ nvcr.io/drive/driveos-sdk/drive-agx-orin-linux-aarch64-sdk-build-x86:6.0.10.0-0009
   ```
   Within the Docker container, execute the following command to install the necessary components for cross-compilation:
   ```bash
   cd /workspace/ && dpkg -i nv-tensorrt-repo-ubuntu2004-cuda11.4-trt8.6.13.3-d6l-cross-ga-20240202_1-1_amd64.deb
   apt install tensorrt-safe-cross-aarch64
   make TARGET=aarch64
   ```
   Running the above commands will generate `/drive/bin/FB-OCC_trt_plugin_aarch.so`, which will be used for TensorRT engine creation in subsequent steps.

   
3. **ONNX Export**  

   First, refer to the [FB-BEV Repository Installation Guide](docs/install.md) for detailed installation instructions for the FB-OCC repository.

   Next, run the following command to generate the ONNX file for FB-OCC. 

   *(Note: Real data samples must be used, and the dataset path thus must be correctly set to avoid errors during ONNX model creation.)*
   ```bash
   python deployment/pth2onnx.py occupancy_configs/fb_occ/fbocc-r50-cbgs_depth_16f_16x4_20e_trt.py --trt_path <path_to_TensorRT> --trt_plugin_path <path_to_TensorRT_plugIn>
   ```
   Running the command above will generate a `create_engine.sh` file. This shell script executes a `trtexec` command with the specified inputs and additional TensorRT configuration flags.

 
4. **TensorRT Engine Creation**  
   Run the following command to create the TensorRT engine:
   ```bash
   sh create_engine.sh 
   ```
   *To create an FP16 model, include the `--fp16` flag along other options in the `trtexec` command.*

   When the command executes successfully, the TensorRT engine will be saved at `data/onnx/fbocc-r50-cbgs_depth_16f_16x4_20e_trt.engine`.
   
   *(Note: Real data samples must be used when creating the engine, as the model utilizes dynamic input sizes for multiple inputs. Ensure the dataset path for inputs (e.g. \*.dat) is correctly configured to prevent errors during TensorRT engine creation.)*

5. **Validation / Inference**  
   To validate the accuracy of the generated TensorRT engine, run the following command:

   ```bash
   python tools/test.py occupancy_configs/fb_occ/fbocc-r50-cbgs_depth_16f_16x4_20e_trt.py ckpts/fbocc-r50-cbgs_depth_16f_16x4_20e.pth --trt_engine <path_to_TensorRT_engine>
   ```
   
   Replace `<path_to_TensorRT_engine>` with the appropriate path to your TensorRT engine. For example, you can use `data/onnx/fbocc-r50-cbgs_depth_16f_16x4_20e_trt.engine`.
