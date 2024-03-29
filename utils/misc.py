#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import math
import numpy as np
import os
import json
import torch.multiprocessing as multiprocessing
import torch.distributed as distributed
from datetime import datetime
import psutil
import torch
from matplotlib import pyplot as plt
from torch import nn
from fvcore.nn.flop_count import flop_count
from iopath.common.file_io import g_pathmgr
import S2TFNet.utils.logging as logging
import S2TFNet.utils.multiprocessing as mpu
from S2TFNet.models.batchnorm_helper import SubBatchNorm3d


logger = logging.get_logger(__name__)


def check_nan_losses(loss):
    """
    Determine whether the loss is NaN (not a number).
    Args:
        loss (loss): loss to check whether is NaN.
    """
    if math.isnan(loss):
        raise RuntimeError("ERROR: Got NaN losses {}".format(datetime.now()))


def params_count(model):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    return np.sum([p.numel() for p in model.parameters()]).item()


def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (GB).
    """
    mem_usage_bytes = torch.cuda.max_memory_allocated()
    return mem_usage_bytes / 1024 ** 3


def cpu_mem_usage():
    """
    Compute the system memory (RAM) usage for the current device (GB).
    Returns:
        usage (float): used memory (GB).
        total (float): total memory (GB).
    """
    vram = psutil.virtual_memory()
    usage = (vram.total - vram.available) / 1024 ** 3
    total = vram.total / 1024 ** 3

    return usage, total


def _get_model_analysis_input(cfg, is_train):
    """
    Return a dummy input for model analysis with batch size 1. The input is
        used for analyzing the model (counting flops and activations etc.).
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        is_train (bool): if True, return the input for training. Otherwise,
            return the input for testing.

    Returns:
        inputs: the input for model analysis.
    """
    rgb_dimension = 3
    if is_train:
        input_tensors = torch.rand(1, rgb_dimension, 8, 224, 224,).cuda(non_blocking=True)
    else:
        if cfg.NUM_GPUS >= 1:
            input_tensors = torch.rand(1, rgb_dimension, 8, 256, 256, ).cuda(non_blocking=True)
        else:
            input_tensors = torch.rand(1, rgb_dimension, 8, 256, 256, )
    # model_inputs = pack_pathway_output(cfg, input_tensors)
    # for i in range(len(model_inputs)):
    #     model_inputs[i] = model_inputs[i].unsqueeze(0).cuda(non_blocking=True)
    #
    # # If detection is enabled, count flops for one proposal.
    inputs = (input_tensors,)
    # inputs = (model_inputs,)
    # if cfg.DETECTION.ENABLE:
    #     bbox = torch.tensor([[0, 0, 1.0, 0, 1.0]])
    #     bbox = bbox.cuda()
    #     inputs = (model_inputs, bbox)
    # else:
    #     inputs = (model_inputs,)
    return inputs


def get_flop_stats(model, cfg, is_train):
    """
    Compute the gflops for the current model given the config.
    Args:
        model (model): model to compute the flop counts.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        is_train (bool): if True, compute flops for training. Otherwise,
            compute flops for testing.

    Returns:
        float: the total number of gflops of the given model.
    """
    inputs = _get_model_analysis_input(cfg, is_train)
    gflop_dict, _ = flop_count(model, inputs)
    gflops = sum(gflop_dict.values())
    return gflops
#
#
# def get_activation_stats(model, cfg, is_train):
#     """
#     Compute the activation count (mega) for the current model given the config.
#     Args:
#         model (model): model to compute the activation counts.
#         cfg (CfgNode): configs. Details can be found in
#             slowfast/config/defaults.py
#         is_train (bool): if True, compute activation for training. Otherwise,
#             compute activation for testing.
#
#     Returns:
#         float: the total number of activation (mega) of the given model.
#     """
#     inputs = _get_model_analysis_input(cfg, is_train)
#     activation_dict, _ = activation_count(model, inputs)
#     activation = sum(activation_dict.values())
#     return activation

def is_eval_epoch(cfg, cur_epoch, multigrid_schedule):
    """
    Determine if the model should be evaluated at the current epoch.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        cur_epoch (int): current epoch.
        multigrid_schedule (List): schedule for multigrid training.
    """
    if cur_epoch + 1 == cfg.SOLVER.MAX_EPOCH:
        return True
    if multigrid_schedule is not None:
        prev_epoch = 0
        for s in multigrid_schedule:
            if cur_epoch < s[-1]:
                period = max(
                    (s[-1] - prev_epoch) // cfg.MULTIGRID.EVAL_FREQ + 1, 1
                )
                return (s[-1] - 1 - cur_epoch) % period == 0
            prev_epoch = s[-1]

    return (cur_epoch + 1) % cfg.TRAIN.EVAL_PERIOD == 0


def plot_input(tensor, bboxes=(), texts=(), path="./tmp_vis.png"):
    """
    Plot the input tensor with the optional bounding box and save it to disk.
    Args:
        tensor (tensor): a tensor with shape of `NxCxHxW`.
        bboxes (tuple): bounding boxes with format of [[x, y, h, w]].
        texts (tuple): a tuple of string to plot.
        path (str): path to the image to save to.
    """
    tensor = tensor - tensor.min()
    tensor = tensor / tensor.max()
    f, ax = plt.subplots(nrows=1, ncols=tensor.shape[0], figsize=(50, 20))
    for i in range(tensor.shape[0]):
        ax[i].axis("off")
        ax[i].imshow(tensor[i].permute(1, 2, 0))
        # ax[1][0].axis('off')
        if bboxes is not None and len(bboxes) > i:
            for box in bboxes[i]:
                x1, y1, x2, y2 = box
                ax[i].vlines(x1, y1, y2, colors="g", linestyles="solid")
                ax[i].vlines(x2, y1, y2, colors="g", linestyles="solid")
                ax[i].hlines(y1, x1, x2, colors="g", linestyles="solid")
                ax[i].hlines(y2, x1, x2, colors="g", linestyles="solid")

        if texts is not None and len(texts) > i:
            ax[i].text(0, 0, texts[i])
    f.savefig(path)


def frozen_bn_stats(model):
    """
    Set all the bn layers to eval mode.
    Args:
        model (model): model to set bn layers to eval mode.
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eval()


def aggregate_sub_bn_stats(module):
    """
    Recursively find all SubBN modules and aggregate sub-BN stats.
    Args:
        module (nn.Module)
    Returns:
        count (int): number of SubBN module found.
    """
    count = 0
    for child in module.children():
        if isinstance(child, SubBatchNorm3d):
            child.aggregate_stats()
            count += 1
        else:
            count += aggregate_sub_bn_stats(child)
    return count


def launch_job(cfg, init_method, func, daemon=False):
    """
    Run 'func' on one or more GPUs, specified in cfg
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        init_method (str): initialization method to launch the job with multiple
            devices.
        func (function): job to run on GPU(s)
        daemon (bool): The spawned processes’ daemon flag. If set to True,
            daemonic processes will be created
    """
    if cfg.NUM_GPUS > 1:
        multiprocessing.spawn(
            mpu.run,
            nprocs=cfg.NUM_GPUS,
            args=(
                cfg.NUM_GPUS,
                func,
                init_method,
                cfg.SHARD_ID,
                cfg.NUM_SHARDS,
                cfg.DIST_BACKEND,
                cfg,
            ),
            daemon=daemon,
        )
    else:
        func(cfg=cfg)


    # if cfg.NUM_GPUS > 1:
    #     distributed.init_process_group(cfg.DIST_BACKEND, init_method)
    #     local_rank = torch.distributed.get_rank()
    #     torch.cuda.set_device(local_rank)
    #     device = torch.device("cuda", local_rank)
    #     func(cfg, device)
    # else:
    #     func(cfg=cfg)


def get_class_names(path, parent_path=None, subset_path=None):
    """
    Read json file with entries {classname: index} and return
    an array of class names in order.
    If parent_path is provided, load and map all children to their ids.
    Args:
        path (str): path to class ids json file.
            File must be in the format {"class1": id1, "class2": id2, ...}
        parent_path (Optional[str]): path to parent-child json file.
            File must be in the format {"parent1": ["child1", "child2", ...], ...}
        subset_path (Optional[str]): path to text file containing a subset
            of class names, separated by newline characters.
    Returns:
        class_names (list of strs): list of class names.
        class_parents (dict): a dictionary where key is the name of the parent class
            and value is a list of ids of the children classes.
        subset_ids (list of ints): list of ids of the classes provided in the
            subset file.
    """
    try:
        with g_pathmgr.open(path, "r") as f:
            class2idx = json.load(f)
    except Exception as err:
        print("Fail to load file from {} with error {}".format(path, err))
        return

    max_key = max(class2idx.values())
    class_names = [None] * (max_key + 1)

    for k, i in class2idx.items():
        class_names[i] = k

    class_parent = None
    if parent_path is not None and parent_path != "":
        try:
            with g_pathmgr.open(parent_path, "r") as f:
                d_parent = json.load(f)
        except EnvironmentError as err:
            print(
                "Fail to load file from {} with error {}".format(
                    parent_path, err
                )
            )
            return
        class_parent = {}
        for parent, children in d_parent.items():
            indices = [
                class2idx[c] for c in children if class2idx.get(c) is not None
            ]
            class_parent[parent] = indices

    subset_ids = None
    if subset_path is not None and subset_path != "":
        try:
            with g_pathmgr.open(subset_path, "r") as f:
                subset = f.read().split("\n")
                subset_ids = [
                    class2idx[name]
                    for name in subset
                    if class2idx.get(name) is not None
                ]
        except EnvironmentError as err:
            print(
                "Fail to load file from {} with error {}".format(
                    subset_path, err
                )
            )
            return

    return class_names, class_parent, subset_ids
