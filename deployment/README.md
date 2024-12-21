# FB-OCC TensorRT Deployment on NVIDIA Drive Platform


This section provides the workflow to deploy  **FB-OCC** on the NVIDIA DRIVE platform using **TensorRT** from model export to inference using TensorRT. It includes all necessary components to streamline the process from model export to execution on TensorRT for NVIDIA DRIVE deployments.

## Occupancy Prediction on NuScenes dataset

   The model [configuration](../occupancy_configs/fb_occ/fbocc-r50-cbgs_depth_16f_16x4_20e_trt.py) to export ONNX file for TensorRT inference is based on the [original configuration](../occupancy_configs/fb_occ/fbocc-r50-cbgs_depth_16f_16x4_20e.py) with modifications.

   - Input resolution: 6 cameras with resolution 256 × 704, forming an input tensor of size 6 × 3 × 256 × 704.
   - Backbone: ResNet-50, consistent with the original configuration.
   - Latency benchmarks on **NVIDIA DRIVE Orin** are measured with NuScenes validation samples.



      |    Model  |    Framework                  | Precision     | mIoU              |  Latency (ms)   |
      |-----------|-----------|---------------|-------------------|--------------------------------------|
      | FB-OCC|[PyTorch](https://github.com/NVlabs/FB-BEV/tree/main?tab=readme-ov-file#model-zoo)       | FP32          | 39.10             | 767.85                                   |
      | FB-OCC|TensorRT      | FP32          | 38.90             | 197.89                               |
      | FB-OCC|TensorRT      | FP16          | 38.86             | 138.62                               |



## Generate ONNX file and save input data

   Before exporting, we assume [the Installation Guide](../docs/install.md) was followed and FB-OCC environment was properly set up.
   

1. **Add functions for exporting ONNX file with custom ops**
   
   FB-OCC uses operations that are not natively supported by TensorRT, including `GridSample3D`, `BevPoolv2`, and `Multi-Scale Deformable Attention`. Therefore, the ONNX file must first be exported with the custom operations, allowing TensorRT to process them as plugins later. A third party [repository](https://github.com/DerryHub/BEVFormer_tensorrt) provides similar functionalities and can be leveraged for FB-OCC:

   Clone the repository and check out the specific commit for compatibility:
   ```bash
   # Clone the BEVFormer_tensorrt repository
   git clone https://github.com/DerryHub/BEVFormer_tensorrt
   # Checkout the specific commit for compatibility
   cd BEVFormer_tensorrt
   git checkout 303d314
   ```

   Those [scripts](https://github.com/DerryHub/BEVFormer_tensorrt/tree/303d3140c14016047c07f9db73312af364f0dd7c/det2trt/models/functions) shall be copied into your workspace and adjusted for FB-OCC by following those steps:

   ```bash
   # Copy BEVFormer_tensorrt functions to the FB-OCC workspace
   cp /path/to/BEVFormer_tensorrt/det2trt/models/functions/*.py /path/to/FB-BEV/deployment/custom_op_functions/

   # Navigate to the target directory and apply the patch for FB-OCC
   cd /path/to/FB-BEV/deployment/custom_op_functions/
   git apply fb-occ_custom_op_functions.patch
   ```

2. **Export ONNX file and save input data**

   Run the following command to create the ONNX file for FB-OCC:
   ```bash
   python deployment/pth2onnx.py occupancy_configs/fb_occ/fbocc-r50-cbgs_depth_16f_16x4_20e_trt.py
   ```

   The command will generate the ONNX model file at `data/onnx/fbocc-r50-cbgs_depth_16f_16x4_20e_trt.onnx`.

   This script also dumps input data at `data/trt_inputs/` which will be needed later to create the TensorRT engine.


## TensorRT Plugin Cross-Compilation for DRIVE Orin on x86 host

   This model is to be deployed on NVIDIA DRIVE Orin with TensorRT 8.6.13.3, which can be downloaded from [NVIDIA DRIVE site](https://developer.nvidia.com/drive/downloads). 
   Please refer to [DRIVE AGX SDK Developer Program](https://developer.nvidia.com/drive/agx-sdk-program) for access to the SDKs.
   

1. **Modify plugin implementation**

   To compile the plugins for the required custom operations, apply the provided patch to adapt the plugins for FB-OCC:

   ```bash
   # Navigate to the BEVFormer_tensorrt directory
   cd /path/to/BEVFormer_tensorrt

   # Apply the patch for TensorRT plugins
   git apply /path/to/FB-BEV/deployment/plugins/fb-occ_trt_plugin.patch
   ```
      
   The patch updates the plugin files in the `TensorRT/` directory to align with FB-OCC requirements and ensure compatibility with TensorRT.
         

2. **Set Up the Environment**

   We recommend using the NVIDIA DRIVE docker image with a pre-configured environment for cross-compilation.

   Launch the docker with the following command:
   ```bash
   docker run --gpus all -it --network=host --rm \
     -v /your/path/to/BEVFormer_tensorrt/:/BEVFormer_tensorrt \
     nvcr.io/drive/driveos-sdk/drive-agx-orin-linux-aarch64-sdk-build-x86:6.0.10.0-0009
   ```

3. **Cross-compile plugins**

   Inside the Docker container, execute the following commands to install the necessary components and build the plugins:   
   ```bash
   cd /BEVFormer_tensorrt/TensorRT/
   make TARGET=aarch64
   ```

   After compilation, the plugin file will be generated at `/drive/bin/aarch64/fb-occ_trt_plugin_aarch64.so`. 
   This file will be used in subsequent steps to create the TensorRT engine.

   
## Build TensorRT Engine on DRIVE Orin

For more details, please refer to [Installation Guide](https://developer.nvidia.com/docs/drive/drive-os/6.0.10/public/drive-os-linux-installation/index.html) to prepare the target.


1. **Prepare and Transfer Required Files**

- ONNX file: `fbocc-r50-cbgs_depth_16f_16x4_20e_trt.onnx`
- Compiled plugin: `fb-occ_trt_plugin_aarch64.so`
- Input data: Files saved in `/path/to/FB-BEV/data/trt_inputs/`
- Shell script: `create_trt_engine.sh` saved in `/path/to/FB-BEV/deployment/`
   

2. **Run the Engine Creation Command** 
   
   Navigate to your workspace on the target platform and execute the following commands to create the TensorRT engine:

   ```bash
   cd /path/to/your/workspace/
   chmod +x create_trt_engine.sh

   # FP32 engine creation 
   ./create_trt_engine.sh \
      --trt_path /usr/src/tensorrt/ \
      --data_dir /path/to/trt_inputs \
      --onnx /path/to/fbocc-r50-cbgs_depth_16f_16x4_20e_trt.onnx \
      --trt_plugin_path /path/to/fb-occ_trt_plugin_aarch64.so \
      --trt_engine_path /path/to/output/fb-occ_trt_engine.plan

   # FP16 engine creation 
   ./create_trt_engine.sh \
      --trt_path /usr/src/tensorrt/ \
      --data_dir /path/to/trt_inputs \
      --onnx /path/to/fbocc-r50-cbgs_depth_16f_16x4_20e_trt.onnx \
      --trt_plugin_path /path/to/fb-occ_trt_plugin_aarch64.so \
      --trt_engine_path /path/to/output/fb-occ_trt_engine_fp16.plan \
      --fp16
   ```

   Upon successful execution, the TensorRT engine will be saved at the specified `/path/to/output/fb-occ_trt_engine.plan`.

   #### **Notes:**

   - **Real Data Requirement:** Ensure real data samples are available and properly configured (e.g., .dat files) to avoid errors during engine creation. The model uses dynamic input sizes for multiple inputs.
   - **Dataset Configuration:** Confirm that the dataset paths for the input files are correctly set up to ensure smooth engine creation.

## TensorRT Engine Evaluation on DRIVE Orin

   The process involves preparing data on an x86 host, running inference on the Orin platform, and returning to the x86 host to evaluate the results and validate the TensorRT engine.

   1. **Preprocess Test Data on x86 Host** 
   
      Prepare all test data on the x86 host by preprocessing it into `.dat` files. Define `--save_dir` as the path to save the preprocessed data.

      ```bash
      cd /path/to/FB-BEV/
      python deployment/eval_orin/preprocess_samples.py \
         occupancy_configs/fb_occ/fbocc-r50-cbgs_depth_16f_16x4_20e_trt.py \
         --save_dir /path/to/preprocessed_data
      ```

   2. **Perform TensorRT Inference on DRIVE Orin**
   
      Mount the preprocessed data and workspace while flashing to DRIVE Orin. 

      Run the shell script in the Docker container to perform TensorRT inference, saving outputs to the `--data_dir` for evaluation.

      ```bash
      cd /path/to/FB-BEV/
      ./deployment/eval_orin/run_data_trt.sh \
         --data_dir /path/to/preprocessed_data \
         --trt_plugin_path /path/to/fb-occ_trt_plugin_aarch64.so \
         --trt_engine_path /path/to/fbocc-r50-cbgs_depth_16f_16x4_20e_trt_orin.engine
      ```

   3. **Transfer Outputs to x86 and Evaluate**
      
      
      Transfer the inference outputs from Orin to the x86 host, postprocess them, and compute accuracy metrics to validate the TensorRT engine.

      To evaluate the TensorRT engine's accuracy, execute the following command:

      ```bash
      cd /path/to/FB-BEV/
      python tools/test.py \
      occupancy_configs/fb_occ/fbocc-r50-cbgs_depth_16f_16x4_20e_trt.py \
      ckpts/fbocc-r50-cbgs_depth_16f_16x4_20e.pth \
      --target_eval \
      --data_dir /path/to/preprocessed_data
      ```
