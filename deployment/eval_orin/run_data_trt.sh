#!/bin/bash

# Parse arguments
DATA_DIR="path/to/data"
TRT_PATH="/usr/src/tensorrt"
TRT_PLUGIN_PATH="fb-occ_trt_plugin_aarch64.so"
TRT_ENGINE_PATH="model.engine"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --data_dir) DATA_DIR="$2"; shift ;;
        --trt_path) TRT_PATH="$2"; shift ;;
        --trt_plugin_path) TRT_PLUGIN_PATH="$2"; shift ;;
        --trt_engine_path) TRT_ENGINE_PATH="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Install numpy if not already installed
if ! python3 -c "import numpy" &> /dev/null; then
    echo "Installing numpy..."
    sudo apt-get update
    sudo apt-get install -y python3-pip
    pip3 install numpy
else
    echo "numpy is already installed."
fi

# Initialize history_bev.dat with zeros
HISTORY_BEV_PATH="${DATA_DIR}/history_bev.dat"
python3 - <<END
import numpy as np
import os
history_bev = np.zeros([1, 1280, 8, 100, 100], dtype=np.float32)
history_bev.tofile('${HISTORY_BEV_PATH}')
print(f"Initialized history_bev.dat at {HISTORY_BEV_PATH}")
END

# Iterate over preprocessed data directories
for idx_dir in "$DATA_DIR"/*; do
    if [ -d "$idx_dir" ]; then
        idx=$(basename "$idx_dir")

        # Build the `trtexec` command for the current data index
        TRTEXEC_CMD="${TRT_PATH}/bin/trtexec"

        PLUGIN_FLAG="--plugins=${TRT_PLUGIN_PATH}"
        ENGINE_FLAG="--loadEngine=${TRT_ENGINE_PATH}"

        # Specify output file path
        OUTPUT_FILE="${idx_dir}/output_.json"

        # Prepare inputs for `trtexec`
        INPUTS_FLAG="--loadInputs=imgs:${idx_dir}/imgs.dat,mlp_input:${idx_dir}/mlp_input.dat,ranks_depth:${idx_dir}/ranks_depth.dat,ranks_feat:${idx_dir}/ranks_feat.dat,ranks_bev:${idx_dir}/ranks_bev.dat,interval_starts:${idx_dir}/interval_starts.dat,interval_lengths:${idx_dir}/interval_lengths.dat,ref_2d:${idx_dir}/ref_2d.dat,reference_points_cam:${idx_dir}/reference_points_cam.dat,per_cam_mask_list:${idx_dir}/per_cam_mask_list.dat,indexes:${idx_dir}/indexes.dat,queries_rebatch:${idx_dir}/queries_rebatch.dat,reference_points_rebatch:${idx_dir}/reference_points_rebatch.dat,bev_query_depth_rebatch:${idx_dir}/bev_query_depth_rebatch.dat,start_of_sequence:${idx_dir}/start_of_sequence.dat,grid:${idx_dir}/grid.dat,history_bev:${HISTORY_BEV_PATH},history_seq_ids:${idx_dir}/history_seq_ids.dat,history_sweep_time:${idx_dir}/history_sweep_time.dat"

        # Add output path flag
        OUTPUT_FLAG="--exportOutput=${OUTPUT_FILE}"

        # Combine all parts into the final command
        FINAL_CMD="${TRTEXEC_CMD} ${PLUGIN_FLAG} ${ENGINE_FLAG} ${INPUTS_FLAG} ${OUTPUT_FLAG}"

        # Execute the command
        echo "Executing command for index ${idx}: $FINAL_CMD"
        eval "$FINAL_CMD"

        # Update history_bev.dat based on trtexec output
        if [ -f "${OUTPUT_FILE}" ]; then
            python3 - <<END
import numpy as np
import json
import os

# Load current history_bev
history_bev_path = "${HISTORY_BEV_PATH}"
history_bev = np.load(history_bev_path)

# Load new output from trtexec
output_file = "${OUTPUT_FILE}"
with open(output_file, 'r') as f:
    trt_output = json.load(f)

# Update history_bev with new values
new_history_bev = np.array(trt_output[1]['values'], dtype=np.float32)
history_bev[:] = new_history_bev

# Save the updated history_bev
np.save(history_bev_path, history_bev)
print(f"Updated history_bev.dat after index ${idx}")
END
        else
            echo "Skipping history_bev update for index ${idx}: Output file not found."
        fi
    fi
done

echo "All samples processed successfully."
