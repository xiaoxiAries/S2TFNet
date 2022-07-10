#!/usr/bin/env python3

"""Wrapper to train and test a video classification model."""


import os
import torch
import torch.distributed as distributed
from SaliNet.utils.parser import load_config, parse_args

from train_net import train
from inference import test

# torch.backends.cudnn.enabled = False


def main():
    """
    Main function to spawn the train and test
    process.
    """
    args = parse_args()
    cfg = load_config(args)
    # Perform training.
    if cfg.TRAIN.ENABLE:
        # launch_job(cfg=cfg, init_method=args.init_method, func=train)
        if cfg.NUM_GPUS > 1:
            distributed.init_process_group(cfg.DIST_BACKEND, args.init_method)
            local_rank = torch.distributed.get_rank()
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            train(cfg, device, local_rank)
    if cfg.TEST.ENABLE:
        # launch_job(cfg=cfg, init_method=args.init_method, func=train)
        if cfg.NUM_GPUS > 1:
            distributed.init_process_group(cfg.DIST_BACKEND, args.init_method)
            local_rank = torch.distributed.get_rank()
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            test(cfg, device, local_rank)
    # # Perform multi-clip testing.
    # if cfg.TEST.ENABLE:
    #     launch_job(cfg=cfg, init_method=args.init_method, func=test)


if __name__ == "__main__":
    main()
