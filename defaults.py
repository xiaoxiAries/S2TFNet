#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Configs."""
from fvcore.common.config import CfgNode

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()


# ---------------------------------------------------------------------------- #
# Batch norm options
# ---------------------------------------------------------------------------- #
_C.BN = CfgNode()

# BN epsilon.
_C.BN.EPSILON = 1e-5

# BN momentum.
_C.BN.MOMENTUM = 0.1

# Precise BN stats.
_C.BN.USE_PRECISE_STATS = False

# Number of samples use to compute precise bn.
_C.BN.NUM_BATCHES_PRECISE = 200

# Weight decay value that applies on BN.*
_C.BN.WEIGHT_DECAY = 0.0

# Norm type, options include `batchnorm`, `sub_batchnorm`, `sync_batchnorm`*
_C.BN.NORM_TYPE = "batchnorm"

# Parameter for SubBatchNorm, where it splits the batch dimension into
# NUM_SPLITS splits, and run BN on each of them separately independently.*
_C.BN.NUM_SPLITS = 1

# Parameter for NaiveSyncBatchNorm3d, where the stats across `NUM_SYNC_DEVICES`
# devices will be synchronized.*
_C.BN.NUM_SYNC_DEVICES = 1


# ---------------------------------------------------------------------------- #
# Training options.
# ---------------------------------------------------------------------------- #
_C.TRAIN = CfgNode()

# If True Train the model, else skip training.
_C.TRAIN.ENABLE = True

# Dataset.*
_C.TRAIN.DATASET = "Kinetics"

# Total mini-batch size.*
_C.TRAIN.BATCH_SIZE = 16

# Evaluate model on test data every eval period epochs.
_C.TRAIN.EVAL_PERIOD = 1

# Save model checkpoint every checkpoint period epochs.*
_C.TRAIN.CHECKPOINT_PERIOD = 1

# Resume training from the latest checkpoint in the output directory.*
_C.TRAIN.AUTO_RESUME = True

# Path to the checkpoint to load the initial weight.*
_C.TRAIN.CHECKPOINT_FILE_PATH = ""

# Checkpoint types include `caffe2` or `pytorch`.*
_C.TRAIN.CHECKPOINT_TYPE = "pytorch"

# If True, perform inflation when loading checkpoint.*
_C.TRAIN.CHECKPOINT_INFLATE = False


# -----------------------------------------------------------------------------
# Model options
# -----------------------------------------------------------------------------
_C.MODEL = CfgNode()

# Model name:resnet10/18/34/50/101/152/200*
_C.MODEL.MODEL_NAME = "resnet10"

# Shortcut type of resnet.*
_C.MODEL.RESNET_SHORTCUT = 'B'

# The number of classes to predict for the model.*
_C.MODEL.NUM_CLASSES = 400

# Loss function.*
_C.MODEL.LOSS_FUNC = "cross_entropy"

# Dropout rate before final projection in the backbone.
# _C.MODEL.DROPOUT_RATE = 0.5


# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------
_C.DATA = CfgNode()

# The json path of the dataset.*
_C.DATA.PATH_TO_DATA_DIR = "/home/lxx/dataset/Kinetics/kinetics-400.json"

# The separator used between path and label.
_C.DATA.PATH_LABEL_SEPARATOR = " "

# Video root path.*
_C.DATA.PATH_PREFIX = "/home/lxx/dataset/Kinetics/"

# The spatial crop size of the input clip.*
_C.DATA.CROP_SIZE = 160

# The number of frames of the input clip.*
_C.DATA.NUM_ALL_FRAMES = 16

# The video sampling rate of the input clip.*
_C.DATA.SAMPLING_RATE = 1

# Clip numbers.*
_C.DATA.CLIP_NUM = 1

# The mean value of the video raw pixels across the R G B channels.*
_C.DATA.MEAN = [0.45, 0.45, 0.45]
# # List of input frame channel dimensions.
#
# _C.DATA.INPUT_CHANNEL_NUM = [3, 3]

# The std value of the video raw pixels across the R G B channels.*
_C.DATA.STD = [0.225, 0.225, 0.225]

# Whether spatial transform*
_C.DATA.SPATIAL_TRANSFORM = True

# Whether temporal transform*
_C.DATA.TEMPORAL_TRANSFORM = True

# Whether target transform*
_C.DATA.TARGET_TRANSFORM = True

# The spatial augmentation jitter scales for training.*
_C.DATA.TRAIN_JITTER_SCALES = [240, 320]

# Input videos may has different fps, convert it to the target video fps before
# frame sampling.
_C.DATA.TARGET_FPS = 30

# Decoding backend, options include `pyav` or `torchvision`
_C.DATA.DECODING_BACKEND = "pyav"

# if True, sample uniformly in [1 / max_scale, 1 / min_scale] and take a
# reciprocal to get the scale. If False, take a uniform sample from
# [min_scale, max_scale].*
_C.DATA.INV_UNIFORM_SAMPLE = False

# If True, perform random horizontal flip on the video frames during training.*
_C.DATA.RANDOM_FLIP = True

# If True, calculdate the map as metric.*
_C.DATA.MULTI_LABEL = False

# Method to perform the ensemble, options include "sum" and "max".
_C.DATA.ENSEMBLE_METHOD = "sum"

# If True, revert the default input channel (RBG <-> BGR).*
_C.DATA.REVERSE_INPUT_CHANNEL = False


# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.SOLVER = CfgNode()

# Base learning rate.*
_C.SOLVER.BASE_LR = 0.1

# Learning rate policy (see utils/lr_policy.py for options and examples).*
_C.SOLVER.LR_POLICY = "steps_with_relative_lrs"

# Exponential decay factor.*
_C.SOLVER.GAMMA = 0.1

# Step size for 'exp' and 'cos' policies (in epochs).
_C.SOLVER.STEP_SIZE = 1

# Steps for 'steps_' policies (in epochs).*
_C.SOLVER.STEPS = [0, 94, 154, 196]

# Learning rates for 'steps_' policies.
_C.SOLVER.LRS = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]

# Maximal number of epochs.*
_C.SOLVER.MAX_EPOCH = 300

# Momentum.*
_C.SOLVER.MOMENTUM = 0.9

# Momentum dampening.*
_C.SOLVER.DAMPENING = 0.0

# Nesterov momentum.*
_C.SOLVER.NESTEROV = True

# L2 regularization.*
_C.SOLVER.WEIGHT_DECAY = 1e-4

# Start the warm up from SOLVER.BASE_LR * SOLVER.WARMUP_FACTOR.
_C.SOLVER.WARMUP_FACTOR = 0.1

# Gradually warm up the SOLVER.BASE_LR over this number of epochs.*
_C.SOLVER.WARMUP_EPOCHS = 34.0

# The start learning rate of the warm up.*
_C.SOLVER.WARMUP_START_LR = 0.01

# Optimization method.*
_C.SOLVER.OPTIMIZING_METHOD = "sgd"


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

# Number of GPUs to use (applies to both training and testing).*
_C.NUM_GPUS = 2

# Whether setting GPU for yourself.*
# _C.GPUS_SETTING_SELF = False

# If setting is True, set the id of GPUS.*
# _C.GPUS_NAME = 0

# Number of machine to use for the job.*
_C.NUM_SHARDS = 1

# The index of the current machine.*
_C.SHARD_ID = 0

# Output basedir.*
_C.OUTPUT_DIR = "./tmp"

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries.*
_C.RNG_SEED = 1

# Log period in iters.*
_C.LOG_PERIOD = 10

# If True, log the model info.*
_C.LOG_MODEL_INFO = True

# Distributed backend.*
_C.DIST_BACKEND = "nccl"

# ---------------------------------------------------------------------------- #
# Benchmark options
# ---------------------------------------------------------------------------- #
_C.BENCHMARK = CfgNode()

# Number of epochs for data loading benchmark.*
_C.BENCHMARK.NUM_EPOCHS = 5

# Log period in iters for data loading benchmark.*
_C.BENCHMARK.LOG_PERIOD = 100

# If True, shuffle dataloader for epoch during benchmark.*
_C.BENCHMARK.SHUFFLE = True


# ---------------------------------------------------------------------------- #
# Common train/test data loader options
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CfgNode()

# Number of data loader workers per training process.*
_C.DATA_LOADER.NUM_WORKERS = 8

# Load data to pinned host memory.*
_C.DATA_LOADER.PIN_MEMORY = True

# Enable multi thread decoding.
_C.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE = False


# ---------------------------------------------------------------------------- #
# Testing options
# ---------------------------------------------------------------------------- #
_C.TEST = CfgNode()

# If True test the model, else skip the testing.
_C.TEST.ENABLE = False

# Dataset for testing.
_C.TEST.DATASET = "Kinetics_inference"

# Total mini-batch size
_C.TEST.BATCH_SIZE = 8

# Path to the checkpoint to load the initial weight.
_C.TEST.CHECKPOINT_FILE_PATH = ""

# Number of clips to sample from a video uniformly for aggregating the
# prediction results.
_C.TEST.NUM_ENSEMBLE_VIEWS = 10
_C.TEST.NUM_ALL_FRAMES = 8
_C.TEST.SAMPLING_RATE = 8

# Checkpoint types include `caffe2` or `pytorch`.
_C.TEST.CHECKPOINT_TYPE = "pytorch"


# ---------------------------------------------------------------------------- #
# Multigrid training options
# See https://arxiv.org/abs/1912.00998 for details about multigrid training.
# ---------------------------------------------------------------------------- #
_C.MULTIGRID = CfgNode()

# Multigrid training allows us to train for more epochs with fewer iterations.
# This hyperparameter specifies how many times more epochs to train.
# The default setting in paper trains for 1.5x more epochs than baseline.*
_C.MULTIGRID.EPOCH_FACTOR = 1.5

# Enable short cycles.*
_C.MULTIGRID.SHORT_CYCLE = True

# Short cycle additional spatial dimensions relative to the default crop size.*
_C.MULTIGRID.SHORT_CYCLE_FACTORS = [0.5, 0.5 ** 0.5]

# Enable long cycles.*
_C.MULTIGRID.LONG_CYCLE = True

# (Temporal, Spatial) dimensions relative to the default shape.*
_C.MULTIGRID.LONG_CYCLE_FACTORS = [(0.25, 0.5 ** 0.5), (0.5, 0.5 ** 0.5), (0.5, 1), (1, 1),]

# While a standard BN computes stats across all examples in a GPU,
# for multigrid training we fix the number of clips to compute BN stats on.
# See https://arxiv.org/abs/1912.00998 for details.*
_C.MULTIGRID.BN_BASE_SIZE = 8

# Multigrid training epochs are not proportional to actual training time or
# computations, so _C.TRAIN.EVAL_PERIOD leads to too frequent or rare
# evaluation. We use a multigrid-specific rule to determine when to evaluate:
# This hyperparameter defines how many times to evaluate a model per long
# cycle shape.*
_C.MULTIGRID.EVAL_FREQ = 3

# No need to specify; Set automatically and used as global variables.*
_C.MULTIGRID.LONG_CYCLE_SAMPLING_RATE = 0

#*
_C.MULTIGRID.DEFAULT_B = 0

#*
_C.MULTIGRID.DEFAULT_T = 0

#*
_C.MULTIGRID.DEFAULT_S = 0

# # -----------------------------------------------------------------------------
# # Tensorboard Visualization Options
# # -----------------------------------------------------------------------------
_C.TENSORBOARD = CfgNode()

# Log to summary writer, this will automatically.
# log loss, lr and metrics during train/eval.
_C.TENSORBOARD.ENABLE = True

# Path to directory for tensorboard logs
# Default to to cfg.OUTPUT_DIR/runs-{cfg.TRAIN.DATASET}.
_C.TENSORBOARD.LOG_DIR = "cfg.OUTPUT_DIR/runs-{cfg.TRAIN.DATASET}"



def _assert_and_infer_cfg(cfg):
    # BN assertions.
    if cfg.BN.USE_PRECISE_STATS:
        assert cfg.BN.NUM_BATCHES_PRECISE >= 0
    # TRAIN assertions.
    assert cfg.TRAIN.CHECKPOINT_TYPE in ["pytorch", "caffe2"]
    assert cfg.TRAIN.BATCH_SIZE % cfg.NUM_GPUS == 0

    # # TEST assertions.
    # assert cfg.TEST.CHECKPOINT_TYPE in ["pytorch", "caffe2"]
    # assert cfg.TEST.BATCH_SIZE % cfg.NUM_GPUS == 0
    # assert cfg.TEST.NUM_SPATIAL_CROPS == 3

    # RESNET assertions.
    # assert cfg.RESNET.NUM_GROUPS > 0
    # assert cfg.RESNET.WIDTH_PER_GROUP > 0
    # assert cfg.RESNET.WIDTH_PER_GROUP % cfg.RESNET.NUM_GROUPS == 0

    # General assertions.
    assert cfg.SHARD_ID < cfg.NUM_SHARDS
    return cfg


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _assert_and_infer_cfg(_C.clone())


