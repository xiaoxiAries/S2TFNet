#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from Non_I3D.datasets.kinetics import Kinetics
from Non_I3D.datasets.ucf101 import UCF101
from Non_I3D.datasets.hmdb51 import HMDB51
from Non_I3D.datasets.kinetics_inference import Kinetics_inference
from Non_I3D.datasets.ucf101_inference import UCF_inference


def build_dataset(dataset_name, cfg, split):
    """
    Build a dataset, defined by `dataset_name`.
    Args:
        dataset_name (str): the name of the dataset to be constructed.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        split (str): the split of the data loader. Options include `train`,
            `val`, and `test`.
    Returns:
        Dataset: a constructed dataset specified by dataset_name.
    """
    # Capitalize the the first letter of the dataset_name since the dataset_name
    # in configs may be in lowercase but the name of dataset class should always
    # start with an uppercase letter.
    if dataset_name == 'Kinetics':
        data = Kinetics(cfg, mode=split)
    elif dataset_name == 'UCF101':
        data = UCF101(cfg, mode=split)
    elif dataset_name == 'HMDB51':
        data = HMDB51(cfg, mode=split)
    elif dataset_name == 'Kinetics_inference':
        data = Kinetics_inference(cfg, mode="validation")
    elif dataset_name == 'UCF_inference' or dataset_name == 'HMDB_inference':
        data = UCF_inference(cfg, mode="validation")
    else:
        data = None
        print('None Data')
    return data


