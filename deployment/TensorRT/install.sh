cd build
cmake .. -DCMAKE_TENSORRT_PATH=/FB-BEV/TensorRT
make -j$(nproc)
make install