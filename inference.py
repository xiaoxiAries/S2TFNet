"""Multi-view test a video classification model."""

import numpy as np
import torch

import S2TFNet.utils.checkpoint as cu
import S2TFNet.utils.distributed as du
import S2TFNet.utils.logging as logging
import S2TFNet.utils.misc as misc
from S2TFNet.datasets import loader
from S2TFNet.models.build import build_model
from S2TFNet.utils.meters import TestMeter

logger = logging.get_logger(__name__)


@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Enable eval mode.
    model.eval()
    test_meter.iter_tic()

    for cur_iter, data in enumerate(test_loader):
        # Transfer the data to the current GPU device.
        inputs, labels, clip_labels = data
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)

        # Transfer the data to the current GPU device.
        labels = labels.cuda()
        clip_labels = clip_labels.cuda()

        # Perform the forward pass.
        preds = model(inputs)

        # Gather all the predictions across all the devices to perform ensemble.
        if cfg.NUM_GPUS > 1:
            preds, labels, clip_labels = du.all_gather([preds[0], labels, clip_labels])

        test_meter.iter_toc()
        # Update and log stats.
        test_meter.update_stats(preds.detach().cpu(), labels.detach().cpu(), clip_labels.detach().cpu())
        test_meter.log_iter_stats(cur_iter)

        test_meter.iter_tic()

    # Log epoch stats and print the final testing results.
    test_meter.finalize_metrics()
    test_meter.reset()


def test(cfg, device, local_rank):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Test with config:")
    logger.info(cfg)

    # Build the video model and print model statistics.
    model = build_model(cfg, device, local_rank)

    # Load a checkpoint to test if applicable.
    if cfg.TEST.CHECKPOINT_FILE_PATH != "":
        cu.load_checkpoint(cfg.TEST.CHECKPOINT_FILE_PATH, model, cfg.NUM_GPUS > 1, None, inflation=False,
                           convert_from_caffe2=cfg.TEST.CHECKPOINT_TYPE == "caffe2")
    elif cu.has_checkpoint(cfg.OUTPUT_DIR):
        last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
        cu.load_checkpoint(last_checkpoint, model, cfg.NUM_GPUS > 1)
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        # If no checkpoint found in TEST.CHECKPOINT_FILE_PATH or in the current
        # checkpoint folder, try to load checkpint from
        # TRAIN.CHECKPOINT_FILE_PATH and test it.
        cu.load_checkpoint(
            cfg.TRAIN.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            None,
            inflation=False,
            convert_from_caffe2=cfg.TRAIN.CHECKPOINT_TYPE == "caffe2",
        )
    else:
        # raise NotImplementedError("Unknown way to load checkpoint.")
        logger.info("Testing with random initialization. Only for debugging.")

    # Create video testing loaders.
    test_loader = loader.construct_loader(cfg, "testing")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    assert (len(test_loader.dataset) % cfg.TEST.NUM_ENSEMBLE_VIEWS == 0)
    # Create meters for multi-view testing.
    test_meter = TestMeter(len(test_loader.dataset) // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROP),
                           cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROP,
                           cfg.MODEL.NUM_CLASSES, len(test_loader), cfg.DATA.MULTI_LABEL, cfg.DATA.ENSEMBLE_METHOD)

    # # Perform multi-view test on the entire dataset.
    perform_test(test_loader, model, test_meter, cfg)

