#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import random
import Non_I3D.utils.logging as logging
import json
# import cv2
import math
import imageio
import numpy as np
import copy
import torch.utils.data

from . import utils as utils
# cv2.setNumThreads(0)
# cv2.ocl.setUseOpenCL(False)

logger = logging.get_logger(__name__)


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))
    return value


def video_loader(video_dir_path, frame_indices):
    video = []
    video_all = imageio.get_reader(video_dir_path, 'ffmpeg')
    for idx in frame_indices:
        frame = video_all.get_data(idx).astype(np.float64)
        video.append(frame)
    if len(video) == 0:
        print("No Data in:", video_dir_path)
    while len(video) < len(frame_indices):
        video.append(frame)
    return video


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []
    frame_nums = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            if subset == 'testing':
                video_names.append('test/{}'.format(key))
            else:
                label = value['annotations']['label']
                video_names.append('{}/{}'.format(label, key))
                annotations.append(value['annotations'])
                frame_nums.append(value['frame_num'])

    return video_names, annotations, frame_nums


def get_valid_frames(n_frames):

    if n_frames > 250:
        n_frames = n_frames - 40  ## delete the invalid frame

    elif (n_frames > 150) and (n_frames <= 250):
        n_frames = n_frames - 30

    else:
        n_frames = n_frames - 5
    return n_frames


def generate_loop_indice(n_frames, cfg):
    out = []
    if cfg.DATA.CLIP_NUM == 1:
        need_frames = cfg.DATA.NUM_ALL_FRAMES * cfg.DATA.SAMPLING_RATE
        if need_frames <= n_frames - 1:
            sample_start = random.randint(1, n_frames - need_frames)
            sample_end = sample_start + need_frames
            out = list(range(sample_start, sample_end, cfg.DATA.SAMPLING_RATE))
        else:
            sample_start = 1
            sample_end = n_frames
            out = list(range(sample_start, sample_end, cfg.DATA.SAMPLING_RATE))
            for index in out:
                if len(out) >= cfg.DATA.NUM_ALL_FRAMES:
                    break
                out.append(index)
    elif cfg.DATA.CLIP_NUM > 1 and cfg.DATA.SAMPLING_RATE > 1:
        need_frames = cfg.DATA.CLIP_NUM * cfg.DATA.SAMPLING_RATE + math.ceil(cfg.DATA.NUM_ALL_FRAMES / cfg.DATA.CLIP_NUM)
        if need_frames <= n_frames - 1:
            sample_start = random.randint(1, n_frames - need_frames)
            out = []
            for idx in range(cfg.DATA.CLIP_NUM):
                out_clip = list(range(sample_start + idx * cfg.DATA.SAMPLING_RATE, sample_start + idx * cfg.DATA.SAMPLING_RATE +
                                      math.ceil(cfg.DATA.NUM_ALL_FRAMES / cfg.DATA.CLIP_NUM)))
                out.extend(out_clip)
        else:
            sample_start = 1
            out = []
            for idx in range(cfg.DATA.CLIP_NUM):
                out_clip = list(range(sample_start + idx * cfg.DATA.SAMPLING_RATE, sample_start + idx * cfg.DATA.SAMPLING_RATE
                                      + math.ceil(cfg.DATA.NUM_ALL_FRAMES / cfg.DATA.CLIP_NUM)))
                out.extend(out_clip)
            for index in out:
                if len(out) >= cfg.DATA.NUM_ALL_FRAMES:
                    break
                out.extend(index)
    return out


def make_dataset(root_path, annotation_path, subset, cfg):
    data = load_annotation_data(annotation_path)
    video_names, annotations, frame_nums = get_video_names_and_annotations(data, subset)
    class_to_idx = get_class_labels(data)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    dataset = []
    for i in range(len(video_names)):
        if i % (math.ceil(len(video_names)/5)) == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_names)))

        video_path = os.path.join(root_path, subset, video_names[i])
        if not os.path.exists(video_path):
            continue
        n_frames = get_valid_frames(frame_nums[i])
        if n_frames <= cfg.DATA.NUM_ALL_FRAMES * cfg.DATA.SAMPLING_RATE:
            continue

        begin_t = 1
        end_t = n_frames
        sample = {
            'video': video_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            'video_id': video_names[i].split('/')[1]
        }
        if len(annotations) != 0:
            sample['label'] = class_to_idx[annotations[i]['label']]
        else:
            sample['label'] = -1

        sample['frame_indices'] = generate_loop_indice(n_frames, cfg)
        dataset.append(sample)
    print('finish data loading')
    return dataset, idx_to_class


class Kinetics(torch.utils.data.Dataset):
    """
    Kinetics video loader. Construct the Kinetics video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, cfg, mode):
        """
        Construct the Kinetics video loader with a given csv file. The format of
        the csv file is:
        ```
        path_to_video_1 label_1
        path_to_video_2 label_2
        ...
        path_to_video_N label_N
        ```
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
        """
        # Only support train, val, and test mode.
        assert mode in ["training", "validation"], "Split '{}' not supported for Kinetics".format(mode)
        self.mode = mode
        self.cfg = cfg

        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.

        logger.info("Constructing Kinetics {}...".format(mode))
        self.data, self.class_names = make_dataset(self.cfg.DATA.PATH_PREFIX, self.cfg.DATA.PATH_TO_DATA_DIR, mode, self.cfg)
        self.spatial_transform = self.cfg.DATA.SPATIAL_TRANSFORM
        self.temporal_transform = self.cfg.DATA.TEMPORAL_TRANSFORM
        self.target_transform = self.cfg.DATA.TARGET_TRANSFORM

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """

        # cv2.setNumThreads(0)
        # cv2.ocl.setUseOpenCL(False)

        short_cycle_idx = None
        # When short cycle is used, input index is a tupple.
        if isinstance(index, tuple):
            index, short_cycle_idx = index

        if self.mode in ["training", "validation"]:
            # -1 indicates random sampling.
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            if short_cycle_idx in [0, 1]:
                crop_size = int(round(self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx] * self.cfg.MULTIGRID.DEFAULT_S))
            if self.cfg.MULTIGRID.DEFAULT_S > 0:
                # Decreasing the scale is equivalent to using a larger "span"
                # in a sampling grid.
                min_scale = int(round(float(min_scale) * crop_size / self.cfg.MULTIGRID.DEFAULT_S))


        path = self.data[index]['video']

        frame_indices = self.data[index]['frame_indices']
        # if self.temporal_transform is not None:
        #     frame_indices = self.temporal_transform(frame_indices)
        clip = video_loader(path, frame_indices)

        if isinstance(clip, (list,)):
            for i in range(len(clip)):
                clip[i] = torch.Tensor(clip[i])

        clip = utils.tensor_normalize(clip, self.cfg.DATA.MEAN, self.cfg.DATA.STD)

        clip = torch.stack(clip, 0).permute(3, 0, 1, 2)

        if self.spatial_transform is not None:
            clip = utils.spatial_sampling(clip, spatial_idx=spatial_sample_index, min_scale=min_scale, max_scale=max_scale,
                                          crop_size=crop_size, random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                                          inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,)

        target = self.data[index]
        if self.target_transform is not None:
            target_transform = utils.ClassLabel()
            target = target_transform(target)
        return clip, target

    def __len__(self):
        return len(self.data)
