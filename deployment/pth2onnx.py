import os
import numpy as np
import argparse
import torch
from torch.onnx import OperatorExportTypes

from mmcv import Config
from mmcv.runner import load_checkpoint

import onnx_graphsurgeon as gs
import onnx 

import sys

sys.path.append(".")

from mmdet3d.models.builder import build_model
from mmdet3d.datasets.builder import build_dataloader, build_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Convert PyTorch to ONNX")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("--checkpoint", default='ckpts/fbocc-r50-cbgs_depth_16f_16x4_20e.pth', help="checkpoint file")
    parser.add_argument("--data_dir", default='data/', help="path to save input data for TRT-engine creation")
    parser.add_argument("--onnx_path", default='data/onnx', help="path to save onnx file")
    parser.add_argument("--trt_path", default='TensorRT-8.6.3.1', help="TensorRT tarball path")
    parser.add_argument("--trt_plugin_path", default='TensorRT/lib/libtensorrt.so', help="TensorRT plugin path")
    parser.add_argument("--opset_version", type=int)
    parser.add_argument("--cuda", default=True, type=bool)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    config_file = args.config
    config = Config.fromfile(config_file)

    if hasattr(config, "plugin"):
        import importlib
        if isinstance(config.plugin, list):
            for plu in config.plugin:
                importlib.import_module(plu)
        else:
            importlib.import_module(config.plugin)

    onnx_path = os.path.join(args.onnx_path, os.path.split(args.config)[1].split(".")[0] + ".onnx")
    dataset = build_dataset(cfg=config.data.val)
    loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=1, shuffle=False, dist=False
    )

    # build the dataloader
    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=1, dist=False, shuffle=False)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **config.data.get('test_dataloader', {})
    }
    dataset = build_dataset(config.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    config.model.train_cfg = None
    model = build_model(config.model, test_cfg=config.get('test_cfg'))
    model.forward = model.forward_trt
    
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu', revise_keys=[(r'^module\.', ''), (r'^teacher\.', '')])
    print('ckpt weight file is loaded')

    model.eval()    
    torch.backends.cudnn.enabled = False
    torch.backends.cuda.matmul.allow_tf32 = False
    
    # get a real data sample
    _, data = next(enumerate(data_loader))
    
    inputs = [t for t in data['img_inputs'][0]]
    meta = data['img_metas'][0].data
    seq_group_idx = int(meta[0][0]['sequence_group_idx']), 
    start_seq = meta[0][0]['start_of_sequence']
    curr2prev_rt = meta[0][0]['curr_to_prev_ego_rt']

    curr_to_prev_ego_rt = torch.stack([curr2prev_rt]) # bs = 1
    seq_ids = torch.LongTensor([seq_group_idx])
    start_of_sequence = torch.BoolTensor([start_seq])

    data_path = args.data_dir
    os.makedirs(args.onnx_path, exist_ok=True)
    
    imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = inputs

    # preprocess the inputs
    with torch.no_grad():
        mlp_input = model.prepare_mlp_inputs([t for t in inputs[1:7]])
        ranks_bev, ranks_depth, ranks_feat, interval_starts, interval_lengths = \
            model.prepare_bevpool_inputs([t for t in inputs[1:7]])
        ref_2d, bev_query_depth, reference_points_cam, per_cam_mask_list, indexes, index_len, \
            queries_rebatch, reference_points_rebatch, bev_query_depth_rebatch = \
        model.prepare_bwdproj_inputs([t for t in inputs[1:7]])
        forward_augs = model.generate_forward_augs(bda)
        history_bev = torch.from_numpy(np.zeros([1, 1280, 8, 100, 100]))
        history_seq_ids = seq_ids
        history_forward_augs = forward_augs.clone()
        history_sweep_time = history_bev.new_zeros(history_bev.shape[0], model.history_cat_num) 
        start_of_sequence = torch.BoolTensor([True])
        grid = model.prepare_grid(bda, history_forward_augs, forward_augs, curr_to_prev_ego_rt, start_of_sequence)

    inputs_ = dict(
                    imgs=imgs.detach().cpu().numpy(),
                    mlp_input=mlp_input.detach().cpu().numpy(), 
                    ranks_depth=ranks_depth.detach().cpu().numpy(), 
                    ranks_feat=ranks_feat.detach().cpu().numpy(),
                    ranks_bev=ranks_bev.detach().cpu().numpy(), 
                    interval_starts=interval_starts.detach().cpu().numpy(), 
                    interval_lengths=interval_lengths.detach().cpu().numpy(), 
                    ref_2d=ref_2d.detach().cpu().numpy(), 
                    bev_query_depth=bev_query_depth.detach().cpu().numpy(),
                    reference_points_cam=reference_points_cam.detach().cpu().numpy(), 
                    per_cam_mask_list=per_cam_mask_list.detach().cpu().numpy(),
                    indexes=indexes.detach().cpu().numpy(), 
                    queries_rebatch=queries_rebatch.detach().cpu().numpy(),
                    reference_points_rebatch=reference_points_rebatch.detach().cpu().numpy(), 
                    bev_query_depth_rebatch=bev_query_depth_rebatch.detach().cpu().numpy(),
                    start_of_sequence=start_of_sequence.detach().cpu().numpy(),
                    grid=grid.detach().cpu().numpy(),
                    history_bev=history_bev.detach().cpu().numpy(),
                    history_seq_ids=history_seq_ids.detach().cpu().numpy().astype(np.int32),
                    history_sweep_time=history_sweep_time.detach().cpu().numpy().astype(np.int32)
                    )

    inputs = {}
    input_Str = ""
    for key in inputs_.keys():
        inputs[key] = inputs_[key]
        if isinstance(inputs_[key], np.ndarray):
            inputs_[key].tofile(os.path.join(data_path, key + '.dat'))
            if not key in ['bev_query_depth']:
                input_Str += key +":" + str(os.path.join(data_path, key + '.dat')) + ","
            inputs[key] = torch.from_numpy(inputs_[key])
        if args.cuda: 
            inputs[key] = inputs[key].cuda()
        assert isinstance(inputs[key], torch.Tensor) or isinstance(inputs[key][0], torch.Tensor)
        
    input_names = list(inputs.keys())
    inputs = [inputs[key] for key in input_names]

    # define outputs
    output_names=['pred_occupancy', 'output_history_bev', 'output_history_seq_ids', 'output_history_sweep_time']

    if args.cuda:
        model.eval().cuda()

    with torch.no_grad():
        torch.onnx.export(
            model,
            inputs,
            onnx_path,
            opset_version=16,
            input_names=input_names,
            output_names=output_names,
            operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH,
            export_params=True,
            do_constant_folding=False,
            verbose=True,
            keep_initializers_as_inputs=True,
            dynamic_axes={'ranks_bev':{0:'ranks_bev_shape_0'}, 
                          'ranks_depth':{0:'ranks_depth_shape_0'},
                          'ranks_feat':{0:'ranks_feat_shape_0'},
                          'interval_starts':{0:'interval_starts_shape_0'},
                          'interval_lengths':{0:'interval_lengths_shape_0'},
                          'indexes':{2:'indexes_shape_0'},
                          'queries_rebatch':{2:'queries_rebatch_shape_0'}, 
                          'reference_points_rebatch':{2:'reference_points_rebatch_shape_0'},
                          'bev_query_depth_rebatch':{2:'bev_query_depth_rebatch_shape_0'}})
    print('ONNX generated and saved at %s' % (onnx_path))
    convert_Reshape_node_allowzero(onnx_path)
    generate_trt_shell_script(onnx_path, args.trt_path, args.trt_plugin_path, input_Str, inputs_)

def convert_Reshape_node_allowzero(onnx_path):
    graph = gs.import_onnx(onnx.load(onnx_path))
    for node in graph.nodes:
        if node.op == "Reshape":
            node.attrs["allowzero"] = 1
    onnx.save(gs.export_onnx(graph), onnx_path)
    return graph

def generate_trt_shell_script(onnx_path, trt_path, trt_plugin_path, input_str, inputs_):
    """
    Generates a shell script 'create_engine.sh' that runs a TensorRT command to create an engine.
    """
    # Define the TensorRT engine path
    trt_engine_path = onnx_path.split('.onnx')[0] + '.engine'

    # Construct the base TensorRT command
    cmd = os.path.join(trt_path, 'bin/trtexec') + \
          f" --onnx={onnx_path} --plugins={trt_plugin_path} --saveEngine={trt_engine_path}"

    # Clean up the input_str if it ends with a comma
    if input_str[-1] == ',':
        input_str = input_str[:-1]

    # Add input loading and shape configurations to the command
    cmd += " --loadInputs=" + input_str + \
          f" --optShapes=ranks_bev:{inputs_['ranks_bev'].shape[0]},ranks_depth:{inputs_['ranks_depth'].shape[0]},ranks_feat:{inputs_['ranks_feat'].shape[0]},interval_starts:{inputs_['interval_starts'].shape[0]},interval_lengths:{inputs_['interval_lengths'].shape[0]},indexes:1x6x{inputs_['indexes'].shape[2]},queries_rebatch:1x6x{inputs_['queries_rebatch'].shape[2]}x80,reference_points_rebatch:1x6x{inputs_['reference_points_rebatch'].shape[2]}x4x2,bev_query_depth_rebatch:1x6x{inputs_['bev_query_depth_rebatch'].shape[2]}x4x1 " + \
          f"--maxShapes=ranks_bev:210000,ranks_depth:210000,ranks_feat:210000,interval_starts:55000,interval_lengths:55000,indexes:1x6x4000,queries_rebatch:1x6x4000x80,reference_points_rebatch:1x6x4000x4x2,bev_query_depth_rebatch:1x6x4000x4x1 " + \
          f"--minShapes=ranks_bev:200000,ranks_depth:200000,ranks_feat:200000,interval_starts:50000,interval_lengths:50000,indexes:1x6x1000,queries_rebatch:1x6x1000x80,reference_points_rebatch:1x6x1000x4x2,bev_query_depth_rebatch:1x6x1000sx4x1"

    # Write the command to a shell script file
    script_content = f"#!/bin/bash\n\n{cmd}\n"
    script_filename = "create_engine.sh"

    with open(script_filename, 'w') as script_file:
        script_file.write(script_content)

    print(f"Shell script '{script_filename}' has been created successfully.")

if __name__ == "__main__":
    main()


