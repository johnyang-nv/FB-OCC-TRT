# FB-OCC TensorRT Deployment on NVIDIA Drive Platform


This section provides the workflow to deploy  **FB-OCC** on the NVIDIA DRIVE platform using **TensorRT**, supporting both `FP32` and `FP16` to inference on NVIDIA DRIVE Orin. It includes all necessary components to streamline the process from model export to execution on TensorRT for NVIDIA DRIVE deployments.
<div align="center">

|                      | Precision     | mIoU              |  Latency (ms) on NVIDIA DRIVE Orin   |
|----------------------|---------------|-------------------|--------------------------------------|
| [FB-OCC-PyTorch](https://github.com/NVlabs/FB-BEV/tree/main?tab=readme-ov-file#model-zoo)       | FP32          | 39.10             | -                                    |
| FB-OCC-TensorRT      | FP32          | 38.90             | 197.89                               |
| FB-OCC-TensorRT      | FP16          | 38.86             | 138.62                               |


</div>




## ONNX Export  

   Before exporting, we assume [the Installation Guide](docs/install.md) was followed and FB-OCC environment was properly set.

1. **Adapting TensorRT Functions for FB-OCC**
   
   The `trt_functions` from the [BEVFormer_tensorrt plugin](https://github.com/DerryHub/BEVFormer_tensorrt/tree/303d3140c14016047c07f9db73312af364f0dd7c/det2trt/models/functions) shall be copied into your workspace and adjusted for FB-OCC by following those steps:

   ```bash
   # Copy BEVFormer_tensorrt functions to the FB-OCC workspace
   cp /path/to/BEVFormer_tensorrt/det2trt/models/functions/*.py /path/to/FB-BEV/deployment/trt_functions/

   # Navigate to the target directory and apply the patch for FB-OCC
   cd /path/to/FB-BEV/deployment/trt_functions/
   git apply fb-occ_trt_functions.patch
   ```

2. **Generating the ONNX File**

   Run the following command to create the ONNX file for FB-OCC:
   ```bash
   python deployment/pth2onnx.py occupancy_configs/fb_occ/fbocc-r50-cbgs_depth_16f_16x4_20e_trt.py
   ```

   The command will generate the ONNX model file at `data/onnx/fbocc-r50-cbgs_depth_16f_16x4_20e_trt.onnx`.

   This script also dumps input data which will be needed later to create the TensorRT engine.


## TensorRT Plugin Cross-Compilation for DRIVE Orin Linux on x86 host

   FB-OCC uses operations that are not natively supported by TensorRT, including `GridSample3D`, `BevPoolv2`, and `Multi-Scale Deformable Attention`. Those operations will be compiled as TensorRT plugins.
   
   ### Steps to Prepare and Build TensorRT Plugins

   1. **Clone and Modify BEVFormer_tensorrt**

      To adapt existing TensorRT plugins for FB-OCC, start with the [BEVFormer_tensorrt](https://github.com/DerryHub/BEVFormer_tensorrt) repository. Clone and modify the repository as follows:

      ```bash
      # Clone the BEVFormer_tensorrt repository
      git clone https://github.com/DerryHub/BEVFormer_tensorrt
      # Checkout the specific commit for compatibility
      cd BEVFormer_tensorrt
      git checkout 303d314
      ```

      Apply the provided patch file to modify the TensorRT plugins for FB-OCC compatibility:

      ```bash
      git apply /path/to/FB-BEV/deployment/plugins/fb-occ_trt_plugin.patch
      ```
      The patch adjusts key plugin files to incorporate FB-OCC-specific operations.
   

2. **Set Up the Environment**

   This model is to be deployed on NVIDIA DRIVE Orin with TensorRT 8.6.13.3. To get access to this version of TensorRT, please refer to details on the [NVIDIA DRIVE site](https://developer.nvidia.com/drive/downloads).    

   First, download the `nv-tensorrt-repo-ubuntu2004-cuda11.4-trt8.6.13.3-d6l-cross-ga-20240202_1-1_amd64.deb` debian package to your workspace `/path/to/BEVFormer_tensorrt/` from [Poduct Information Delivery](https://apps.nvidia.com/PID/ContentGroup/Detail/1948?FromLocation=CL) with your NVONLINE account. 

   Launch the Drive OS Linux Docker container with `BEVFormer_tensorrt` mounted::
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

   
## Running TensorRT Engine Creation on DRIVE Orin

To create the TensorRT engine, confirm that the target platform (e.g., NVIDIA DRIVE Orin) is set up with NVIDIA DRIVE OS. 
Use the [flashing procedures](https://developer.nvidia.com/drive/downloads) to prepare the target system.


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

## TensorRT Engine Evaluation on DRIVE Orin

   The process involves preparing data on an x86 host, performing inference on the target platform (Orin), and completing evaluation back on the x86 host. Follow the steps below to validate the TensorRT engine and evaluate its performance:

   1. **Preprocess Test Data on x86 Host** 
   
      Prepare all test data on the x86 host by preprocessing it into `.dat` or `.bin` files. Define `--save_dir` as the path to save the preprocessed data.

      ```bash
      cd /path/to/FB-BEV/
      python deployment/eval_orin/preprocess_samples.py \
         occupancy_configs/fb_occ/fbocc-r50-cbgs_depth_16f_16x4_20e_trt.py \
         --save_dir /path/to/preprocessed_data

      ```

   2. **Perform TensorRT Inference on DRIVE Orin**
   
      Mount the preprocessed data and workspace while flashing to DRIVE Orin. 

      Then, run the shell script to perform TensorRT inference on all preprocessed data within the Docker container. The script saves the outputs back into the `--data_dir` directory for further evaluation.

      ```bash
      cd /path/to/FB-BEV/deployment/
      ./eval_orin/run_data_trt.sh \
         --data_dir /path/to/preprocessed_data \
         --trt_plugin_path /path/to/fb-occ_trt_plugin_aarch64.so \
         --trt_engine_path /path/to/fbocc-r50-cbgs_depth_16f_16x4_20e_trt_orin.engine
      ```

   3. **Transfer Outputs Back to x86 and Evaluate**
      
      After completing inferences for the entire dataset on the Orin platform, transfer the output files back to the x86 host for evaluation. Postprocess the results and compute accuracy metrics to validate the performance of the TensorRT engine.

      To evaluate the TensorRT engine's accuracy, execute the following command:

      ```bash
      cd /path/to/FB-BEV/
      python tools/test.py \
      occupancy_configs/fb_occ/fbocc-r50-cbgs_depth_16f_16x4_20e_trt.py \
      ckpts/fbocc-r50-cbgs_depth_16f_16x4_20e.pth \
      --target_eval \
      --data_dir /path/to/preprocessed_data

      ```
