#!/usr/bin/env python3

"""Model construction functions."""

import torch
from .model import generate_model


def build_model(cfg, device, local_rank):
    """
    Builds the video model.
    Args:
        cfg (configs): configs that contains the hyper-parameters to build the
        backbone. Details can be seen in slowfast/config/defaults.py.
    """

    # Construct the model
    model = generate_model(cfg)
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.parallel.DistributedDataParallel(module=model, device_ids=[local_rank],
                                                          output_device=local_rank)
    return model
