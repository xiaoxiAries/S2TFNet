#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Data loader."""
import torch
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

from SaliNet.datasets.multigrid_helper import ShortCycleBatchSampler

from .build import build_dataset


def construct_loader(cfg, split, is_precise_bn=False):
    """
    Constructs the data loader for the given dataset.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        split (str): the split of the data loader. Options include `train`,
            `val`, and `test`.
    """
    assert split in ["training", "validation", "testing"]
    if split in ["training"]:
        dataset_name = cfg.TRAIN.DATASET
        batch_size = int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS)
        shuffle = True
        drop_last = True
    elif split in ["validation"]:
        dataset_name = cfg.TRAIN.DATASET
        batch_size = int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS)
        shuffle = False
        drop_last = False
    elif split in ["testing"]:
        dataset_name = cfg.TEST.DATASET
        batch_size = int(cfg.TEST.BATCH_SIZE / cfg.NUM_GPUS) if cfg.NUM_GPUS >= 1 else cfg.TENSORBOARD.BATCHSIZE
        shuffle = False
        drop_last = False

    # Construct the dataset

    dataset = build_dataset(dataset_name, cfg, split)


    if cfg.MULTIGRID.SHORT_CYCLE and split in ["training"] and not is_precise_bn:
        # Create a sampler for multi-process training
        sampler = (DistributedSampler(dataset) if cfg.NUM_GPUS > 1 else RandomSampler(dataset))
        batch_sampler = ShortCycleBatchSampler(sampler, batch_size=batch_size, drop_last=drop_last, cfg=cfg)
        # Create a loader
        # loader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler, num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        #                                      pin_memory=cfg.DATA_LOADER.PIN_MEMORY,)
        loader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler, num_workers=cfg.DATA_LOADER.NUM_WORKERS,
                                             pin_memory=cfg.DATA_LOADER.PIN_MEMORY)

    else:
        # Create a sampler for multi-process training
        # sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        # loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=cfg.DATA_LOADER.NUM_WORKERS)
        sampler = DistributedSampler(dataset) if cfg.NUM_GPUS > 1 else None
        # Create a loader
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=(False if sampler else shuffle),
                                             sampler=sampler, num_workers=cfg.DATA_LOADER.NUM_WORKERS,
                                             pin_memory=cfg.DATA_LOADER.PIN_MEMORY, drop_last=drop_last)
        # loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=(False if sampler else shuffle),
        #                                      sampler=sampler, num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        #                                      pin_memory=cfg.DATA_LOADER.PIN_MEMORY, drop_last=drop_last,)

    return loader


def shuffle_dataset(loader, cur_epoch):
    """"
    Shuffles the data.
    Args:
        loader (loader): data loader to perform shuffle.
        cur_epoch (int): number of the current epoch.
    """
    sampler = (loader.batch_sampler.sampler if isinstance(loader.batch_sampler, ShortCycleBatchSampler) else loader.sampler)
    assert isinstance( sampler, (RandomSampler, DistributedSampler)), "Sampler type '{}' not supported".format(type(sampler))
    # RandomSampler handles shuffling automatically
    if isinstance(sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        sampler.set_epoch(cur_epoch)
