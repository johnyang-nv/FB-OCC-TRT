# FB-OCC TensorRT Deployment

This repository provides a comprehensive deployment framework for **FB-OCC** using **TensorRT**, supporting both **FP32** and **FP16** inference for optimized performance. It includes all necessary components to streamline the process from model export to efficient execution on TensorRT.

1. **Prerequisites**
   
   Download the TensorRT-8.6.3.1 tarball from the [NVIDIA TensorRT - Get Started page](https://developer.nvidia.com/tensorrt-getting-started). TensorRT is available for free as a binary on multiple platforms or as a container on NVIDIA NGC™. After downloading, follow the installation instructions provided on the page to set up TensorRT on your system.

   ```bash
   tar -xzvf <path_to_your_TensorRT-8.6.3.1-tarball.tar.gz>
   export LD_LIBRARY_PATH=<path_to_your_TensorRT-8.6.3.1>:$LD_LIBRARY_PATH
   ```

2. **TensorRT Plugin Compilation**  

   FB-OCC utilizes operations that are not natively supported in TensorRT, such as `GridSample3D`, `BevPoolv2` and `Multi-Scale Deformable Attention`. These operations must be implemented as custom TensorRT plugins and compiled before using FB-OCC for inference. These plugins extend TensorRT’s capabilities, enabling the model to function as expected during optimization and deployment.

   The TensorRT plugins provided in the [BEVFormer_tensorrt](https://github.com/DerryHub/BEVFormer_tensorrt) repository can be modified to work with FB-OCC by applying the necessary tweaks . Follow the [TensorRT plugin compilation steps in this repository](https://github.com/DerryHub/BEVFormer_tensorrt?tab=readme-ov-file#build-tensorrt-plugins-of-mmdeploy), using the provided patch applied to the TensorRT/ subdirectory. 

   ### Steps to Compile TensorRT Plugins for FB-OCC:

   ```bash
   git clone https://github.com/DerryHub/BEVFormer_tensorrt
   git checkout 303d314
   cd BEVFormer_tensorrt/
   git apply --directory=TensorRT/ <path_to_patch_file>
   ```
   Replace `<path_to_patch_file>` with the path to this `FB-BEV/deployment/plugins` patch file.
   
   ### For x86  with TensorRT-8.6.3.1

   After successfully applying the patch and compiling the TensorRT plugin as instructed in the `BEVFormer_tensorrt` repo, the file `BEVFormer_tensorrt/TensorRT/lib/libtensorrt.so` will be generated and can be used for running inference.

   ### For NVIDIA DRIVE Orin with TensorRT-8.6.13.3

   Cross-compiling is essential when the target platform, such as NVIDIA DRIVE OS Linux, differs from the development environment (x86 host). It allows developers to build ARM-compatible plugins on the x86 host by leveraging its computational power. 
   
   To cross-compile, set up the required toolchains, libraries, and environment variables, then compile the source code to generate binaries for the DRIVE OS Linux platform.


   plugins/FB-OCC_trt_plugin_aarch.patch 
   BEVFormer_tensorrt에다가, 파일 다시 정리 하고 kernels, include, so on. 그리고, deb dpkg

   First, obtain the NGC container with the pre-configured environment.
   docker run --gpus all -it --network=host --rm -v /home/johnyang/BEVFormer_tensorrt/:/BEVFormer_tensorrt/ nvcr.io/drive/driveos-sdk/drive-agx-orin-linux-aarch64-sdk-build-x86:6.0.10.0-0009


   ```bash
   nvcr.io/drive/driveos-sdk/drive-agx-orin-linux-aarch64-sdk-build-x86:6.0.10.0-0009

   ```





   
3. **ONNX Export**  

   First, refer to the [FB-BEV Repository Installation Guide](docs/install.md) for detailed installation instructions for the FB-OCC repository.

   Next, run the following command to generate the ONNX file for FB-OCC.
   *(Note: Real data samples must be used, and the dataset path thus must be correctly set to avoid errors during ONNX model creation.)*
   ```bash
   python deployment/pth2onnx.py deployment/occupancy_trt_configs/fbocc-r50-cbgs_depth_16f_16x4_20e_trt.py --trt_path <path_to_TensorRT-8.6.3.1> --trt_plugin_path <path_to_TensorRT_PlugIn>
   ```
   Running the command above will generate a `create_engine.sh` file

 
4. **TensorRT Engine Creation**  
   Run the following command to create the TensorRT engine:
   ```bash
   sh create_engine.sh 
   ```
   When the `TRTEXEC` command executes successfully, the TensorRT engine will be saved at `data/onnx/fbocc-r50-cbgs_depth_16f_16x4_20e_trt.engine`.
   
   *(Note: Real data samples must be used when creating the engine, as the model utilizes dynamic input sizes for multiple inputs. Ensure the dataset path for inputs (e.g. \*.dat) is correctly configured to prevent errors during TensorRT engine creation.)*

5. **Inference Script**  
   To validate the accuracy of the generated TensorRT engine, run the following command:

   ```bash
   python tools/test.py ./occupancy_configs/fb_occ/fbocc-r50-cbgs_depth_16f_16x4_20e_trt.py ckpts/fbocc-r50-cbgs_depth_16f_16x4_20e.pth --trt_engine <path_to_TensorRT_engine>
   ```
   
   Replace `<path_to_TensorRT_engine>` with the appropriate path to your TensorRT engine. For example, you can use `data/onnx/fbocc-r50-cbgs_depth_16f_16x4_20e_trt.engine`.
