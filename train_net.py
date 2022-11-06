#!/usr/bin/env python3

"""Train a video classification model."""
import numpy as np
import pprint
import torch
import S2TFNet.models.losses as losses
import S2TFNet.models.optimizer as optim
import S2TFNet.utils.logging as logging
import S2TFNet.utils.checkpoint as cu
import S2TFNet.utils.distributed as du
import S2TFNet.utils.metrics as metrics
import S2TFNet.utils.misc as misc
import S2TFNet.utils.tensorboard_vis as tb
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats
from S2TFNet.datasets import loader
from S2TFNet.models.build import build_model
from S2TFNet.utils.meters import TrainMeter, ValMeter
from S2TFNet.utils.multigrid import MultigridSchedule

logger = logging.get_logger(__name__)


def train_epoch(train_loader, model, optimizer, train_meter, cur_epoch, cfg, writer=None):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)
    for cur_iter, data in enumerate(train_loader):
        # Transfer the data to the current GPU device.
        inputs, labels = data
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)

        preds = model(inputs)

        # Explicitly declare reduction to mean and compute the loss.
        loss_fun = losses.loss_func(cfg)
        # loss = loss_fun(preds, labels)
        # loss_fun = torch.nn.CrossEntropyLoss(reduction="mean")
        loss = loss_fun(preds, labels)
        # check Nan Loss.
        misc.check_nan_losses(loss)

        # Perform the backward pass.
        optimizer.zero_grad()
        loss.backward()
        # Update the parameters.
        optimizer.step()
        top1_err, top5_err = None, None
        if cfg.DATA.MULTI_LABEL:
            # Gather all the predictions across all the devices.
            if cfg.NUM_GPUS > 1:
                [loss] = du.all_reduce([loss])
            loss = loss.item()
        else:
            # Compute the errors.
            num_topks_correct = metrics.topks_correct(preds[0], labels, (1, 5))
            top1_err, top5_err = [(1.0 - x / preds[0].size(0)) * 100.0 for x in num_topks_correct]

            # Gather all the predictions across all the devices.
            if cfg.NUM_GPUS > 1:
                loss, top1_err, top5_err = du.all_reduce([loss, top1_err, top5_err])

            # Copy the stats from GPU to CPU (sync point).
            loss, top1_err, top5_err = (loss.item(), top1_err.item(), top5_err.item(),)

        train_meter.iter_toc()
        # Update and log stats.
        train_meter.update_stats(top1_err, top5_err, loss, lr, inputs[0].size(0) * cfg.NUM_GPUS)
        # write to tensorboard format if available.
        if writer is not None:
            writer.add_scalars(
                {
                    "Train/loss": loss,
                    "Train/lr": lr,
                    "Train/Top1_err": top1_err,
                    "Train/Top5_err": top5_err,
                },
                global_step=data_size * cur_epoch + cur_iter,
            )

        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer=None):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()

    for cur_iter, (inputs, labels) in enumerate(val_loader):
        # Transferthe data to the current GPU device.

        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        preds = model(inputs)

        if cfg.DATA.MULTI_LABEL:
            if cfg.NUM_GPUS > 1:
                preds, labels = du.all_gather([preds[0], labels])
        else:
            # Compute the errors.
            num_topks_correct = metrics.topks_correct(preds[0], labels, (1, 5))
            # Combine the errors across the GPUs.
            top1_err, top5_err = [(1.0 - x / preds[0].size(0)) * 100.0 for x in num_topks_correct]
            if cfg.NUM_GPUS > 1:
                top1_err, top5_err = du.all_reduce([top1_err, top5_err])

            # Copy the errors from GPU to CPU (sync point).
            top1_err, top5_err = top1_err.item(), top5_err.item()

            val_meter.iter_toc()
            # Update and log stats.
            val_meter.update_stats(top1_err, top5_err, inputs[0].size(0) * cfg.NUM_GPUS)
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars({"Val/Top1_err": top1_err, "Val/Top5_err": top5_err},
                                   global_step=len(val_loader) * cur_epoch + cur_iter,)

        val_meter.update_predictions(preds[0], labels)

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)
    # write to tensorboard format if available.
    # if writer is not None:
    #     writer.add_scalars({"Val/mAP": val_meter.full_map}, global_step=cur_epoch)

    val_meter.reset()


def build_trainer(cfg, device, local_rank):
    """
    Build training model and its associated tools, including optimizer,
    dataloaders and meters.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    Returns:
        model (nn.Module): training model.
        optimizer (Optimizer): optimizer.
        train_loader (DataLoader): training data loader.
        val_loader (DataLoader): validatoin data loader.
        precise_bn_loader (DataLoader): training data loader for computing
            precise BN.
        train_meter (TrainMeter): tool for measuring training stats.
        val_meter (ValMeter): tool for measuring validation stats.
    """
    # Build the video model and print model statistics.
    model = build_model(cfg, device, local_rank)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, is_train=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "training")
    val_loader = loader.construct_loader(cfg, "validation")
    if cfg.BN.USE_PRECISE_STATS:
        precise_bn_loader = loader.construct_loader(cfg, "training", is_precise_bn=True)
    else:
        precise_bn_loader = None
    # Create meters.
    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    return (model, optimizer, train_loader, val_loader, precise_bn_loader, train_meter, val_meter,)


def calculate_and_update_precise_bn(loader, model, num_iters=200):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
    """

    def _gen_loader():
        for inputs, _, in loader:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)


def train(cfg, device, local_rank):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
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

    # Init multigrid.
    multigrid = None
    if cfg.MULTIGRID.LONG_CYCLE or cfg.MULTIGRID.SHORT_CYCLE:
        multigrid = MultigridSchedule()
        cfg = multigrid.init_multigrid(cfg)
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, _ = multigrid.update_long_cycle(cfg, cur_epoch=0)
    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = build_model(cfg, device, local_rank)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Load a checkpoint to resume training if applicable.
    if cfg.TRAIN.AUTO_RESUME and cu.has_checkpoint(cfg.OUTPUT_DIR):
        last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
        logger.info("Load from last checkpoint, {}.".format(last_checkpoint))
        checkpoint_epoch = cu.load_checkpoint(last_checkpoint, model, cfg.NUM_GPUS > 1, optimizer)
        start_epoch = checkpoint_epoch + 1
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        logger.info("Load from given checkpoint file.")
        checkpoint_epoch = cu.load_checkpoint(cfg.TRAIN.CHECKPOINT_FILE_PATH, model, cfg.NUM_GPUS > 1, optimizer,
                                              inflation=cfg.TRAIN.CHECKPOINT_INFLATE,
                                              convert_from_caffe2=cfg.TRAIN.CHECKPOINT_TYPE == "caffe2",)
        start_epoch = checkpoint_epoch + 1

    elif cfg.TRAIN.PRE_FILE_PATH != "" and cfg.MODEL.NUM_CLASSES == 400:
        logger.info("Load from given pretrained file which has part of the parameters.")
        checkpoint_epoch = cu.load_part_of_resnet_checkpoint(cfg.TRAIN.PRE_FILE_PATH, model, cfg.NUM_GPUS > 1, None)

        # start_epoch = checkpoint_epoch + 1
        start_epoch = 0
    elif cfg.TRAIN.PRE_FILE_PATH != "" and cfg.MODEL.NUM_CLASSES != 400:
        checkpoint_epoch = cu.load_part_of_checkpoint_for_other_dataset(cfg.TRAIN.PRE_FILE_PATH, model, cfg.NUM_GPUS > 1, None)
        start_epoch = checkpoint_epoch
    else:
        start_epoch = 0

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "training")
    val_loader = loader.construct_loader(cfg, "validation")
    if cfg.BN.USE_PRECISE_STATS:
        precise_bn_loader = loader.construct_loader(cfg, "training", is_precise_bn=True)
    else:
        precise_bn_loader = None

    # Create meters.

    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    # # set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE:
        writer = tb.TensorboardWriter(cfg)
    # else:
    #     writer = None
    # writer = tb.TensorboardWriter(cfg)

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, changed = multigrid.update_long_cycle(cfg, cur_epoch)
            if changed:
                (model, optimizer, train_loader, val_loader, precise_bn_loader, train_meter, val_meter,) = build_trainer(cfg, device, local_rank)

                # Load checkpoint.
                if cu.has_checkpoint(cfg.OUTPUT_DIR):
                    last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
                    assert "{:05d}.pyth".format(cur_epoch) in last_checkpoint
                else:
                    last_checkpoint = cfg.TRAIN.CHECKPOINT_FILE_PATH
                logger.info("Load from {}".format(last_checkpoint))
                cu.load_checkpoint(last_checkpoint, model, cfg.NUM_GPUS > 1, optimizer)

        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)
        # Train for one epoch.
        train_epoch(train_loader, model, optimizer, train_meter, cur_epoch, cfg, writer)

        # Compute precise BN stats.
        if cfg.BN.USE_PRECISE_STATS and len(get_bn_modules(model)) > 0:
            calculate_and_update_precise_bn(precise_bn_loader, model, min(cfg.BN.NUM_BATCHES_PRECISE, len(precise_bn_loader)),)
        _ = misc.aggregate_sub_bn_stats(model)

        # Save a checkpoint.
        if cu.is_checkpoint_epoch(cfg, cur_epoch, None if multigrid is None else multigrid.schedule):
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, cfg)
        # Evaluate the model on validation set.
        if misc.is_eval_epoch(cfg, cur_epoch, None if multigrid is None else multigrid.schedule):
            eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer)

    if writer is not None:
        writer.close()
