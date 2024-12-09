# FB-OCC TensorRT Deployment

This repository provides a comprehensive deployment framework for **FB-OCC** (Feature-Based Object Classification) using **TensorRT**, supporting both **FP32** and **FP16** inference for optimized performance. It includes all necessary components to streamline the process from model export to efficient execution on TensorRT.

### Key Features:
1. **ONNX Export**  

   Tools and scripts for converting your FB-OCC model into the ONNX format, serving as an intermediate representation for TensorRT optimization.

   First, occupancy_path = 'data/nuscenes/gts' 

2. **Plugin Compilation**  
   Custom TensorRT plugin compilation for advanced operations not natively supported by TensorRT.
   - **TensorRT-8.6.3.1** for x86 inference cases.
   - **TensorRT-8.6.13.3** for **Drive Orin**.

3. **TensorRT Engine Creation**  
   Scripts for building TensorRT engines from the ONNX model, enabling high-performance inference.

4. **Inference Script**  
   A Python script for running inference with the generated TensorRT engine, including accuracy evaluation to validate the deployment pipeline.
