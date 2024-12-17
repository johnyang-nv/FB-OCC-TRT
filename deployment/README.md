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
   cd BEVFormer_tensorrt
   git checkout 303d314
   ```

## ONNX Export  
   First, refer to the [FB-BEV Repository Installation Guide](docs/install.md) for detailed installation instructions for the FB-OCC repository. The guide also includes details about the pre-saved checkpoints located in `ckpts/`.

1. **Adapting TensorRT Functions for FB-OCC**
   
   The `trt_functions` from the [BEVFormer_tensorrt plugin](https://github.com/DerryHub/BEVFormer_tensorrt/tree/303d3140c14016047c07f9db73312af364f0dd7c/det2trt/models/functions) must be copied into the FB-OCC workspace and adjusted for compatibility. Follow these csteps:

   ```bash
   # Copy BEVFormer_tensorrt functions to the FB-OCC workspace
   cp /path/to/BEVFormer_tensorrt/det2trt/models/functions/*.py /path/to/FB-BEV/deployment/trt_functions/

   # Navigate to the target directory and apply the patch for FB-OCC
   cd /path/to/FB-BEV/deployment/trt_functions/
   git apply fb-occ_trt_fn-patch-on-derryhub_fn.patch
   ```

2. **Generating the ONNX File**

   Run the following command to create the ONNX file for FB-OCC:
   ```bash
   python deployment/pth2onnx.py occupancy_configs/fb_occ/fbocc-r50-cbgs_depth_16f_16x4_20e_trt.py
   ```

   The command will generate the ONNX model file at `data/onnx/fbocc-r50-cbgs_depth_16f_16x4_20e_trt.onnx`.

   During execution, the command processes real data samples and saves the input data in the directory `data/trt_inputs/`. 
   These inputs are needed for the next steps, where they will be used to build the TensorRT engine.


## TensorRT Plugin Cross-Compilation for DRIVE OS Linux on x86 host

   Cross-compiling is essential when the target platform, such as NVIDIA DRIVE OS Linux, differs from the development environment (x86 host). It allows developers to build ARM-compatible plugins on the x86 host by leveraging its computational power. 

1. **Apply the Patch**

   To prepare the TensorRT plugins for cross-compilation, apply the `fb-occ_trt_plugin.patch` to the plugin files:

   ```bash
   cd /path/to/BEVFormer_tensorrt/
   git apply /path/to/FB-BEV/deployment/plugins/fb-occ_trt_plugin.patch
   ```   

2. **Set Up the Environment**

   - Download a pre-configured NGC container to streamline the cross-compilation process. Detailed instructions for accessing NGC containers are available in the [NVIDIA DRIVE site](https://developer.nvidia.com/drive/downloads).    
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

   After the compilation is finished, the plugin file will be generated at the following location:: `/drive/bin/aarch64/fb-occ_trt_plugin_aarch64.so`.

   Move the plugin file to your mounted directory for use in the subsequent steps:
   ```bash
   mv /drive/bin/aarch64/fb-occ_trt_plugin_aarch64.so /BEVFormer_tensorrt/
   ```

   The plugin file `fb-occ_trt_plugin_aarch64.so` will be used when creating the TensorRT engine.

   
## Running TensorRT Engine Creation on the Target Platform

Before proceeding with TensorRT engine creation, ensure that the target platform (e.g., NVIDIA DRIVE Orin) is properly set up with NVIDIA DRIVE OS. 
Use [the procedures in this section](https://developer.nvidia.com/docs/drive/drive-os/6.0.10/public/drive-os-linux-installation/common/topics/installation/docker-ngc/setup-drive-os-linux-nvonline.html#ariaid-title5) to flash NVIDIA DRIVE OS Linux to the target system from the Docker container. Following these steps will prepare the target system with the necessary operating system, drivers, and tools.


1. **Prepare and Transfer Required Files**

   Prepare the following files to the target platform:

   - ONNX file: `fbocc-r50-cbgs_depth_16f_16x4_20e_trt.onnx`
   - Compiled plugin: `fb-occ_trt_plugin_aarch64.so`
   - Input data: Files saved in `/path/to/FB-BEV/data/trt_inputs/`
   - Shell script: `create_trt_engine.sh` saved in `/path/to/FB-BEV/deployment/`
   

2. **Run the Engine Creation Command** 
   
   Navigate to your workspace on the target platform and execute the following commands to create the TensorRT engine:

   ```bash
   cd /path/to/your/workspace/
   chmod +x create_trt_engine.sh

   # Example 1: Standard engine creation
   ./create_trt_engine.sh --trt_plugin_path /path/to/fb-occ_trt_plugin_aarch64.so

   # Example 2: Engine creation with FP16 precision
   ./create_trt_engine.sh --trt_plugin_path /path/to/fb-occ_trt_plugin_aarch64.so --fp16

   # Example 3: Custom engine path
   ./create_trt_engine.sh --trt_plugin_path /path/to/fb-occ_trt_plugin_aarch64.so --trt_engine_path /path/to/custom_engine_path

   # Example 4: Custom engine path and custom input data path
   ./create_trt_engine.sh --trt_plugin_path /path/to/fb-occ_trt_plugin_aarch64.so --trt_engine_path /path/to/custom_engine_path --data_dir /path/to/trt_inputs 
   ```

   Upon successful execution, the TensorRT engine will be saved at the specified `<path_to_TensorRT_engine>`.

   #### **Notes:**

   - **Real Data Requirement:** Ensure real data samples are available and properly configured (e.g., .dat files) to avoid errors during engine creation. The model uses dynamic input sizes for multiple inputs.
   - **Dataset Configuration:** Confirm that the dataset paths for the input files are correctly set up to ensure smooth engine creation.

## TensorRT Engine Evaluation on the Target Platform

   The process involves preparing data on an x86 host, performing inference on the target platform (Orin), and completing evaluation back on the x86 host. Follow the steps below to validate the TensorRT engine and evaluate its performance:

   1. **Preprocess Test Data on x86 Host** 
   
      Prepare all test data on the x86 host by preprocessing it into `.dat` or `.bin` files. Define `--save_dir` as the path to save the preprocessed data.

      ```bash
      cd /path/to/FB-BEV/
      python deployment/eval_orin/preprocess_samples.py \
         occupancy_configs/fb_occ/fbocc-r50-cbgs_depth_16f_16x4_20e_trt.py \
         --save_dir /path/to/preprocessed_data

      ```

   2. **Perform TensorRT Inference on NVIDIA DRIVE Orin**
   
      Copy or mount the preprocessed data and workspace onto the target platform while flashing NVIDIA DRIVE OS Linux to the target system. Use [this guide](https://developer.nvidia.com/docs/drive/drive-os/6.0.10/public/drive-os-linux-installation/common/topics/installation/docker-ngc/setup-drive-os-linux-nvonline.html#ariaid-title5) to set up the system within a Docker container.

      Run the shell script to perform TensorRT inference on all preprocessed data within the Docker container. The script saves the outputs back into the `--data_dir` directory for further evaluation.

      ```bash
      cd /path/to/FB-BEV/deployment/
      ./eval_orin/run_data_trt.sh \
         --data_dir /path/to/preprocessed_data \
         --trt_plugin_path /path/to/fb-occ_trt_plugin_aarch64.so \
         --trt_engine_path /path/to/fbocc-r50-cbgs_depth_16f_16x4_20e_trt_orin.engine
      ```

   3. **Transfer Outputs Back to x86 and Evaluate**
      
      Once inference is complete on the Orin platform, transfer the output files to the x86 host for evaluation. Postprocess the results and compute accuracy metrics to validate the performance of the TensorRT engine.

      To evaluate the TensorRT engine's accuracy on the target platform, execute the following command:

      ```bash
      python tools/test.py \
      occupancy_configs/fb_occ/fbocc-r50-cbgs_depth_16f_16x4_20e_trt.py \
      ckpts/fbocc-r50-cbgs_depth_16f_16x4_20e.pth \
      --target_eval \
      --data_dir /path/to/preprocessed_data

      ```

   **Results**

   The TensorRT engine produces results consistent with the PyTorch implementation on NVIDIA DRIVE Orin:
   ```bash
   FP32 Precision: 38.90
   FP16 Precision: 38.86
   ```

   These results align closely with the reproduced PyTorch accuracy of `38.90` from the original implementation.
   TensorRT models demonstrate reduced inference latency compared to the PyTorch implementation, with further improvements observed when using FP16 precision.

