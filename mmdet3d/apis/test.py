# Copyright (c) OpenMMLab. All rights reserved.
from os import path as osp

import mmcv
import torch
from mmcv.image import tensor2imgs
import time
from mmdet3d.models import (Base3DDetector, Base3DSegmentor,
                            SingleStageMono3DDetector)


def single_gpu_test_trt(model,
                        data_loader,
                        engine_path, 
                        cfg):
    """Test TensorRT engine inference with single gpu.

    This method tests model with single gpu with a given TensorRT engine. 

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        engine_path (str): The path to TensorRT engine. 

    Returns:
        list[dict]: The prediction results.
    """
    from deployment.utils.trt_infer_utils import run_trt, build_engine
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))

    for idx, data in enumerate(data_loader):
        with torch.no_grad():
            trt_inputs = []
            inputs = [t for t in data['img_inputs'][0]]
            meta = data['img_metas'][0].data
            seq_group_idx = int(meta[0][0]['sequence_group_idx']), 
            start_seq = meta[0][0]['start_of_sequence']
            curr2prev_rt = meta[0][0]['curr_to_prev_ego_rt']

            curr_to_prev_ego_rt = torch.stack([curr2prev_rt]) # bs = 1
            seq_ids = torch.LongTensor([seq_group_idx])
            start_of_sequence = torch.BoolTensor([start_seq])
            imgs, _, _, _, _, _, bda = inputs
            
            mlp_input = model.prepare_mlp_inputs([t for t in inputs[1:7]])
            ranks_bev, ranks_depth, ranks_feat, interval_starts, interval_lengths = \
                model.prepare_bevpool_inputs([t.cuda() for t in inputs[1:7]])
            ref_2d, bev_query_depth, reference_points_cam, per_cam_mask_list, indexes, index_len, \
                queries_rebatch, reference_points_rebatch, bev_query_depth_rebatch = \
                model.prepare_bwdproj_inputs([t for t in inputs[1:7]])
            
            forward_augs = model.generate_forward_augs(bda)

            if idx == 0 : 
                history_bev = torch.from_numpy(np.zeros(cfg.output_shapes['output_history_bev']))
                history_seq_ids = seq_ids
                history_forward_augs = forward_augs.clone()
                history_sweep_time = history_bev.new_zeros(history_bev.shape[0], model.history_cat_num) 
                start_of_sequence = torch.BoolTensor([True])
            else: 
                history_sweep_time += 1
                if start_of_sequence.sum() > 0.: 
                    history_sweep_time[start_of_sequence] = 0
                    history_seq_ids[start_of_sequence] = seq_ids[start_of_sequence]
                    history_forward_augs[start_of_sequence] = forward_augs[start_of_sequence]
                    
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
            
            input_shapes = {}
            for id, key in enumerate(inputs_.keys()):
                input_shapes[key] = [element for element in  inputs_[key].shape]
                trt_inputs.append(inputs_[key]) # NUMPY array
            
            engine = build_engine(engine_path)
            trt_outputs, t = run_trt(trt_inputs, engine, imgs.size(0), input_shapes, cfg.output_shapes)
            
            pred_occupancy = torch.from_numpy(trt_outputs['pred_occupancy'])
            history_bev = torch.from_numpy(trt_outputs['output_history_bev'])
            history_seq_ids = torch.from_numpy(trt_outputs['output_history_seq_ids'])
            history_sweep_time = torch.from_numpy(trt_outputs['output_history_sweep_time'])
            
            result_dict_list = model.post_process(pred_occupancy)
            result_dict_list[0]['index'] = meta[0][0]['index']
            results.extend(result_dict_list)

            batch_size = imgs.size(0)
            for _ in range(batch_size):
                prog_bar.update()
    return results

def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    """Test model with single gpu.

    This method tests model with single gpu and gives the 'show' option.
    By setting ``show=True``, it saves the visualization results under
    ``out_dir``.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool, optional): Whether to save viualization results.
            Default: True.
        out_dir (str, optional): The path to save visualization results.
            Default: None.

    Returns:
        list[dict]: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            
        if show:
            # Visualize the results of MMDetection3D model
            # 'show_results' is MMdetection3D visualization API
            models_3d = (Base3DDetector, Base3DSegmentor,
                         SingleStageMono3DDetector)
            if isinstance(model.module, models_3d):
                model.module.show_results(
                    data,
                    result,
                    out_dir=out_dir,
                    show=show,
                    score_thr=show_score_thr)
            # Visualize the results of MMDetection model
            # 'show_result' is MMdetection visualization API
            else:
                batch_size = len(result)
                if batch_size == 1 and isinstance(data['img'][0],
                                                  torch.Tensor):
                    img_tensor = data['img'][0]
                else:
                    img_tensor = data['img'][0].data[0]
                img_metas = data['img_metas'][0].data[0]
                imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
                assert len(imgs) == len(img_metas)

                for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                    h, w, _ = img_meta['img_shape']
                    img_show = img[:h, :w, :]

                    ori_h, ori_w = img_meta['ori_shape'][:-1]
                    img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                    if out_dir:
                        out_file = osp.join(out_dir, img_meta['ori_filename'])
                    else:
                        out_file = None

                    model.module.show_result(
                        img_show,
                        result[i],
                        show=show,
                        out_file=out_file,
                        score_thr=show_score_thr)
        results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results


# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import os.path as osp
import pickle
import shutil
import tempfile
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from mmdet.core import encode_mask_results


import mmcv
import numpy as np
import pycocotools.mask as mask_util

def custom_encode_mask_results(mask_results):
    """Encode bitmap mask to RLE code. Semantic Masks only
    Args:
        mask_results (list | tuple[list]): bitmap mask results.
            In mask scoring rcnn, mask_results is a tuple of (segm_results,
            segm_cls_score).
    Returns:
        list | tuple: RLE encoded mask.
    """
    cls_segms = mask_results
    num_classes = len(cls_segms)
    encoded_mask_results = []
    for i in range(len(cls_segms)):
        encoded_mask_results.append(
            mask_util.encode(
                np.array(
                    cls_segms[i][:, :, np.newaxis], order='F',
                        dtype='uint8'))[0])  # encoded with RLE
    return [encoded_mask_results]

def custom_multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.
    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
    Returns:
        list: The prediction results.
    """
    model.eval()
    bbox_results = []
    mask_results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    have_mask = False
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            # encode mask results
            if isinstance(result, dict):
                if 'bbox_results' in result.keys():
                    bbox_result = result['bbox_results']
                    batch_size = len(result['bbox_results'])
                    bbox_results.extend(bbox_result)
                if 'mask_results' in result.keys() and result['mask_results'] is not None:
                    mask_result = custom_encode_mask_results(result['mask_results'])
                    mask_results.extend(mask_result)
                    have_mask = True
            else:
                batch_size = len(result)
                bbox_results.extend(result)

            #if isinstance(result[0], tuple):
            #    assert False, 'this code is for instance segmentation, which our code will not utilize.'
            #    result = [(bbox_results, encode_mask_results(mask_results))
            #              for bbox_results, mask_results in result]
        if rank == 0:
            
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        bbox_results = collect_results_gpu(bbox_results, len(dataset))
        if have_mask:
            mask_results = collect_results_gpu(mask_results, len(dataset))
        else:
            mask_results = None
    else:
        bbox_results = collect_results_cpu(bbox_results, len(dataset), tmpdir)
        tmpdir = tmpdir+'_mask' if tmpdir is not None else None
        if have_mask:
            mask_results = collect_results_cpu(mask_results, len(dataset), tmpdir)
        else:
            mask_results = None

    if mask_results is None:
        return bbox_results
    return {'bbox_results': bbox_results, 'mask_results': mask_results}




def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    tmpdir = None
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            prefix = str(time.time())[-5:]
            tmpdir = tempfile.mkdtemp(dir='.dist_test', prefix=prefix)
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        '''
        bacause we change the sample of the evaluation stage to make sure that each gpu will handle continuous sample,
        '''
        #for res in zip(*part_list):
        for res in part_list:  
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results #[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    collect_results_cpu(result_part, size)