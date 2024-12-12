import os
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Create a TensorRT Engine")
    parser.add_argument("--data_dir", default="data/trt_inputs", help="Path to the input data for TensorRT engine creation.")
    parser.add_argument("--onnx", default="data/onnx/fbocc-r50-cbgs_depth_16f_16x4_20e_trt.onnx", help="Path to the ONNX file.")
    parser.add_argument("--trt_path", default="/usr/src/tensorrt/", help="Path to the TensorRT installation directory.")
    parser.add_argument("--trt_plugin_path", default="TensorRT/lib/libtensorrt.so", help="Path to the TensorRT plugin library.")
    parser.add_argument("--trt_engine_path", default="data/fbocc-r50-cbgs_depth_16f_16x4_20e_trt.engine", help="Path to save the generated TensorRT engine.")
    parser.add_argument("--fp16", action="store_true", default=False, help="Enable creation of a TensorRT engine with FP16 precision.")
    return parser.parse_args()

def main():
    args = parse_args()

    # Determine the shapes of dynamic axes
    ranks_shape = np.load(os.path.join(args.data_dir, 'ranks_bev.npy')).shape[0]
    interval_shape = np.load(os.path.join(args.data_dir, 'interval_starts.npy')).shape[0]
    indexes_shape = np.load(os.path.join(args.data_dir, 'indexes.npy')).shape[2]

    # Construct base command
    trtexec_cmd = os.path.join(args.trt_path, 'bin/trtexec') + ' --useCudaGraph'
    if args.fp16:
        trtexec_cmd += ' --fp16'

    # ONNX and plugin paths
    onnx_path = f" --onnx={args.onnx}"
    plugin_path = f" --plugins={args.trt_plugin_path}"

    # Save engine path (optional if needed)
    save_engine_path = f" --saveEngine={args.trt_engine_path}"

    # Load inputs and shapes
    load_inputs = (
        f"--loadInputs="
        f"imgs:{args.data_dir}/imgs.dat,"
        f"mlp_input:{args.data_dir}/mlp_input.dat,"
        f"ranks_depth:{args.data_dir}/ranks_depth.dat,"
        f"ranks_feat:{args.data_dir}/ranks_feat.dat,"
        f"ranks_bev:{args.data_dir}/ranks_bev.dat,"
        f"interval_starts:{args.data_dir}/interval_starts.dat,"
        f"interval_lengths:{args.data_dir}/interval_lengths.dat,"
        f"ref_2d:{args.data_dir}/ref_2d.dat,"
        f"reference_points_cam:{args.data_dir}/reference_points_cam.dat,"
        f"per_cam_mask_list:{args.data_dir}/per_cam_mask_list.dat,"
        f"indexes:{args.data_dir}/indexes.dat,"
        f"queries_rebatch:{args.data_dir}/queries_rebatch.dat,"
        f"reference_points_rebatch:{args.data_dir}/reference_points_rebatch.dat,"
        f"bev_query_depth_rebatch:{args.data_dir}/bev_query_depth_rebatch.dat,"
        f"start_of_sequence:{args.data_dir}/start_of_sequence.dat,"
        f"grid:{args.data_dir}/grid.dat,"
        f"history_bev:{args.data_dir}/history_bev.dat,"
        f"history_seq_ids:{args.data_dir}/history_seq_ids.dat,"
        f"history_sweep_time:{args.data_dir}/history_sweep_time.dat"
    )

    opt_shapes = (
        f"--optShapes="
        f"ranks_bev:{ranks_shape},"
        f"ranks_depth:{ranks_shape},"
        f"ranks_feat:{ranks_shape},"
        f"interval_starts:{interval_shape},"
        f"interval_lengths:{interval_shape},"
        f"indexes:1x6x{indexes_shape},"
        f"queries_rebatch:1x6x{indexes_shape}x80,"
        f"reference_points_rebatch:1x6x{indexes_shape}x4x2,"
        f"bev_query_depth_rebatch:1x6x{indexes_shape}x4x1"
    )

    min_shapes = (
        f"--minShapes="
        f"ranks_bev:200000,"
        f"ranks_depth:200000,"
        f"ranks_feat:200000,"
        f"interval_starts:50000,"
        f"interval_lengths:50000,"
        f"indexes:1x6x1000,"
        f"queries_rebatch:1x6x1000x80,"
        f"reference_points_rebatch:1x6x1000x4x2,"
        f"bev_query_depth_rebatch:1x6x1000x4x1"
    )

    max_shapes = (
        f"--maxShapes="
        f"ranks_bev:210000,"
        f"ranks_depth:210000,"
        f"ranks_feat:210000,"
        f"interval_starts:55000,"
        f"interval_lengths:55000,"
        f"indexes:1x6x4000,"
        f"queries_rebatch:1x6x4000x80,"
        f"reference_points_rebatch:1x6x4000x4x2,"
        f"bev_query_depth_rebatch:1x6x4000x4x1"
    )

    # Combine all parts to form the final command
    final_cmd = f"{trtexec_cmd}{onnx_path}{plugin_path}{save_engine_path} {load_inputs} {opt_shapes} {min_shapes} {max_shapes}"

    print("Executing command:", final_cmd)
    
    # Execute the final command
    os.system(final_cmd)


if __name__ == "__main__":
    main()

