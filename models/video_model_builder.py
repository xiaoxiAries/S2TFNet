#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Video models."""

import torch.nn as nn
import torch

import Non_I3D.utils.weight_init_helper as init_helper
from Non_I3D.models.batchnorm_helper import get_norm

from . import resnet_helper, stem_helper
from Non_I3D.models.SSF_TSFP_Module import Sidelayer_3x3x3, Fusion_Temp_For_Mask, TSFP_1, TSFP_2

# Number of blocks for different stages given the model depth.
_MODEL_STAGE_DEPTH = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3)}
_NUM_BLOCK_TEMP_KERNEL = {50:([3], [4], [6], [3]), 101: ([3], [4], [23], [3])}
# Basis of temporal kernel sizes for each of the stage.
_TEMPORAL_KERNEL_BASIS = {
    "i3d": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ]
}

_POOL1 = {
    "i3d": [[2, 1, 1]],
    "i3d_nopool": [[1, 1, 1]],
}


class ResNet(nn.Module):
    """
    ResNet model builder. It builds a ResNet like network backbone without
    lateral connection (C2D, I3D, Slow).

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "SlowFast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf

    Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He.
    "Non-local neural networks."
    https://arxiv.org/pdf/1711.07971.pdf
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(ResNet, self).__init__()
        self.norm_module = get_norm(cfg)
        # self.enable_detection = cfg.DETECTION.ENABLE
        self.num_pathways = 1
        self._construct_network(cfg)
        self.cfg = cfg
        init_helper.init_weights(self, 0.01, True)

    def _construct_network(self, cfg):
        """
        Builds a single pathway ResNet model.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        pool_size = _POOL1['i3d']
        assert len({len(pool_size), self.num_pathways}) == 1

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.MODEL.DEPTH]
        (block_temp_kernel1, block_temp_kernel2, block_temp_kernel3, block_temp_kernel4) = _NUM_BLOCK_TEMP_KERNEL[cfg.MODEL.DEPTH]

        num_groups = 1
        width_per_group = 64
        dim_inner = num_groups * width_per_group

        temp_kernel = _TEMPORAL_KERNEL_BASIS['i3d']

        self.s1 = stem_helper.VideoModelStem(
            dim_in=[3],
            dim_out=[width_per_group],
            kernel=[temp_kernel[0][0] + [7, 7]],
            stride=[[1, 2, 2]],
            padding=[[temp_kernel[0][0][0] // 2, 3, 3]],
            norm_module=self.norm_module,
        )

        self.s2 = resnet_helper.ResStage(
            dim_in=[width_per_group],
            dim_out=[width_per_group * 4],
            dim_inner=[dim_inner],
            temp_kernel_sizes=temp_kernel[1],
            stride=[1],
            num_blocks=[d2],
            num_groups=[num_groups],
            num_block_temp_kernel=block_temp_kernel1,
            nonlocal_inds=[[]],
            nonlocal_group=[1],
            nonlocal_pool=[[1, 2, 2], [1, 2, 2]],
            instantiation='softmax',
            trans_func_name='bottleneck_transform',
            stride_1x1=False,
            inplace_relu=True,
            dilation=[1],
            norm_module=self.norm_module,
        )

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = resnet_helper.ResStage(
            dim_in=[width_per_group * 4],
            dim_out=[width_per_group * 8],
            dim_inner=[dim_inner * 2],
            temp_kernel_sizes=temp_kernel[2],
            stride=[2],
            num_blocks=[d3],
            num_groups=[num_groups],
            num_block_temp_kernel=block_temp_kernel2,
            nonlocal_inds=[[1, 3]],
            nonlocal_group=[1],
            nonlocal_pool=[[1, 2, 2], [1, 2, 2]],
            instantiation='softmax',
            trans_func_name='bottleneck_transform',
            stride_1x1=False,
            inplace_relu=True,
            dilation=[1],
            norm_module=self.norm_module,
        )

        self.s4 = resnet_helper.ResStage(
            dim_in=[width_per_group * 8],
            dim_out=[width_per_group * 16],
            dim_inner=[dim_inner * 4],
            temp_kernel_sizes=temp_kernel[3],
            stride=[2],
            num_blocks=[d4],
            num_groups=[num_groups],
            num_block_temp_kernel=block_temp_kernel3,
            nonlocal_inds=[[1, 3, 5]],
            nonlocal_group=[1],
            nonlocal_pool=[[1, 2, 2], [1, 2, 2]],
            instantiation='softmax',
            trans_func_name='bottleneck_transform',
            stride_1x1=False,
            inplace_relu=True,
            dilation=[1],
            norm_module=self.norm_module,
        )

        self.s5 = resnet_helper.ResStage(
            dim_in=[width_per_group * 16],
            dim_out=[width_per_group * 32],
            dim_inner=[dim_inner * 8],
            temp_kernel_sizes=temp_kernel[4],
            stride=[2],
            num_blocks=[d5],
            num_groups=[num_groups],
            num_block_temp_kernel=block_temp_kernel4,
            nonlocal_inds=[[]],
            nonlocal_group=[1],
            nonlocal_pool=[[1, 2, 2], [1, 2, 2]],
            instantiation='softmax',
            trans_func_name='bottleneck_transform',
            stride_1x1=False,
            inplace_relu=True,
            dilation=[1],
            norm_module=self.norm_module,
        )
        if cfg.MODEL.SSF_ENABLE:
            self.sidelayer1 = Sidelayer_3x3x3(in_channels=256, out_channels=1)
            self.sidelayer2 = Sidelayer_3x3x3(in_channels=512, out_channels=1)
            self.sidelayer3 = Sidelayer_3x3x3(in_channels=1024, out_channels=1)
            self.bn_one = nn.BatchNorm3d(num_features=1, eps=1e-5, momentum=0.1)
            self.average_pooling = nn.AvgPool3d(kernel_size=2, padding=0, stride=2)
            self.sigmoid = nn.Sigmoid()
            self.fusion_temp_mask = Fusion_Temp_For_Mask(in_channels=4, out_channels=1)
            self.average_pooling_S = nn.AvgPool3d(kernel_size=(1, 2, 2), padding=0, stride=(1, 2, 2))

        if cfg.MODEL.SSF_ENABLE and cfg.MODEL.TSFP_ENABLE is False:
            self.average_pooling_S = nn.AvgPool3d(kernel_size=(1, 2, 2), padding=0, stride=(1, 2, 2))

        if cfg.MODEL.TSFP_ENABLE:
            self.TSPF_layer_1 = TSFP_1(in_planes=1, out_planes=1)
            self.TSPF_layer_2 = TSFP_2(in_planes=1, out_planes=1)
            # self.TSPF_layer_1 = TSFP(in_planes=1, out_planes=1)

        self.head = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(p=cfg.MODEL.DROPOUT_RATE)

        self.fc = nn.Linear(width_per_group * 32, cfg.MODEL.NUM_CLASSES)

    def forward(self, x):
        if not self.cfg.MODEL.SSF_ENABLE:
            x = self.s1(x)
            x = self.s2(x)
            for pathway in range(self.num_pathways):
                pool = getattr(self, "pathway{}_pool".format(pathway))
                x[pathway] = pool(x[pathway])
            x = self.s3(x)
            x = self.s4(x)
            x = self.s5(x)
            x = self.head(x[0])
            x = x.view(-1, self.fc.in_features)
            x = self.fc(self.dropout(x))

        elif self.cfg.MODEL.TSFP_ENABLE:
            x = self.s1(x)
            x = self.s2(x)
            for pathway in range(self.num_pathways):
                pool = getattr(self, "pathway{}_pool".format(pathway))
                x[pathway] = pool(x[pathway])
            x_side1 = self.sidelayer1(x)
            x_side1 = self.bn_one(x_side1[0])
            x_side1 = self.sigmoid(x_side1)
            x_side1 = self.TSPF_layer_1(x_side1)
            x = self.s3(x)
            x_side2 = self.sidelayer2(x)
            x_side2 = self.bn_one(x_side2[0])
            x_side2 = self.sigmoid(x_side2)
            size_x = x_side1[0].size()
            x_side1[0] = x_side1[0].chunk(size_x[2], 2)
            x_side2 = x_side2.chunk(size_x[2], 2)
            x_side2_concat = torch.cat((x_side1[0][0], x_side2[0], x_side1[0][1], x_side2[1], x_side1[0][2], x_side2[2],
                                        x_side1[0][3], x_side2[3]), 2)
            x_side2_concat = self.TSPF_layer_2(x_side2_concat)
            x = self.s4(x)
            x_side3 = self.sidelayer3(x)
            x_side3 = self.bn_one(x_side3[0])
            x_side3 = self.sigmoid(x_side3)
            size_x = x_side2_concat[0].size()
            x_side2_concat[0] = x_side2_concat[0].chunk(size_x[2], 2)
            x_side3 = x_side3.chunk(size_x[2], 2)
            x_side3_concat = torch.cat((x_side2_concat[0][0], x_side3[0], x_side2_concat[0][1], x_side3[1],
                                        x_side2_concat[0][2], x_side3[2], x_side2_concat[0][3], x_side3[3]), 2)

            # x_side2_concat[0] = self.average_pooling(x_sidfe2_concat[0])
            # x = self.s4(x)
            # x_side3 = self.sidelayer3(x)
            # x_side3 = self.bn_one(x_side3[0])
            # x_side3 = self.sigmoid(x_side3)
            # size_x = x_side2_concat[0].size()
            # x_side2_concat[0] = x_side2_concat[0].chunk(size_x[2], 2)
            # x_side3 = x_side3.chunk(size_x[2], 2)
            # x_side3_concat = torch.cat((x_side2_concat[0][0], x_side3[0], x_side2_concat[0][1], x_side3[1],
            #                             x_side2_concat[0][2], x_side3[2], x_side2_concat[0][3], x_side3[3]), 2)


            x_side3_concat = self.average_pooling(x_side3_concat)
            x = self.s5(x)
            F = self.fusion_temp_mask(x_side3_concat)
            x = torch.mul(F, x[0])
            x = self.head(x)
            x = x.view(-1, self.fc.in_features)
            x = self.fc(self.dropout(x))

        else:
            x = self.s1(x)
            x = self.s2(x)
            for pathway in range(self.num_pathways):
                pool = getattr(self, "pathway{}_pool".format(pathway))
                x[pathway] = pool(x[pathway])
            x_side1 = self.sidelayer1(x)
            x_side1 = self.bn_one(x_side1[0])
            x_side1 = self.sigmoid(x_side1)
            x_side1 = self.average_pooling_S(x_side1)
            x = self.s3(x)
            x_side2 = self.sidelayer2(x)
            x_side2 = self.bn_one(x_side2[0])
            x_side2 = self.sigmoid(x_side2)
            size_x = x_side1.size()
            x_side1 = x_side1.chunk(size_x[2], 2)
            x_side2 = x_side2.chunk(size_x[2], 2)
            x_side2_concat = torch.cat((x_side1[0], x_side2[0], x_side1[1], x_side2[1], x_side1[2], x_side2[2],
                                        x_side1[3], x_side2[3]), 2)
            x_side2_concat = self.average_pooling(x_side2_concat)
            x = self.s4(x)
            x_side3 = self.sidelayer3(x)
            x_side3 = self.bn_one(x_side3[0])
            x_side3 = self.sigmoid(x_side3)
            size_x = x_side2_concat.size()
            x_side2_concat = x_side2_concat.chunk(size_x[2], 2)
            x_side3 = x_side3.chunk(size_x[2], 2)
            x_side3_concat = torch.cat((x_side2_concat[0], x_side3[0], x_side2_concat[1], x_side3[1],
                                        x_side2_concat[2], x_side3[2], x_side2_concat[3], x_side3[3]), 2)
            x_side3_concat = self.average_pooling(x_side3_concat)

            # x_side3_concat = self.average_pooling_S(x_side3)

            x = self.s5(x)
            F = self.fusion_temp_mask(x_side3_concat)
            x = torch.mul(F, x[0])
            x = self.head(x)
            x = x.view(-1, self.fc.in_features)
            x = self.fc(self.dropout(x))


        if self.cfg.MODEL.LOSS_FUNC == "CE_L1_L2":
            return [x, F, x_side1, x_side2_concat]
        elif self.cfg.MODEL.LOSS_FUNC == "CE_L1":
            return [x, F]
        elif self.cfg.MODEL.LOSS_FUNC == "cross_entropy":
            return [x]



        #
        #
        # x = self.s1(x)
        # x = self.s2(x)
        # for pathway in range(self.num_pathways):
        #     pool = getattr(self, "pathway{}_pool".format(pathway))
        #     x[pathway] = pool(x[pathway])
        #
        # if self.cfg.MODEL.SSF_ENABLE:
        #     x_side1 = self.sidelayer1(x)
        #     x_side1 = self.bn_one(x_side1[0])
        #     x_side1 = self.sigmoid(x_side1)
        #
        #     if self.cfg.MODEL.TSFP_ENABLE:
        #         x_side1 = self.TSPF_layer_1(x_side1)
        #     else:
        #         x_side1 = self.average_pooling_S(x_side1)
        #
        # x = self.s3(x)
        #
        # if self.cfg.MODEL.SSF_ENABLE:
        #     x_side2 = self.sidelayer2(x)
        #     x_side2 = self.bn_one(x_side2[0])
        #     x_side2 = self.sigmoid(x_side2)
        #
        #     if self.cfg.MODEL.TSFP_ENABLE:
        #         size_x = x_side1.size()
        #         x_side1 = x_side1.chunk(size_x[2], 2)
        #         x_side2 = x_side2.chunk(size_x[2], 2)
        #         x_side2_concat = torch.cat((x_side1[0], x_side2[0], x_side1[1], x_side2[1], x_side1[2], x_side2[2],
        #                                     x_side1[3], x_side2[3]), 2)
        #         x_side2_concat, x_side2_concat_loss = self.TSPF_layer_2(x_side2_concat)
        #     else:
        #         size_x = x_side1.size()
        #         x_side1 = x_side1.chunk(size_x[2], 2)
        #         x_side2 = x_side2.chunk(size_x[2], 2)
        #         x_side2_concat = torch.cat((x_side1[0], x_side2[0], x_side1[1], x_side2[1], x_side1[2], x_side2[2],
        #                                     x_side1[3], x_side2[3]), 2)
        #         x_side2_concat = self.average_pooling(x_side2_concat)
        #
        #
        # x = self.s4(x)
        #
        # if self.cfg.MODEL.SSF_ENABLE:
        #     x_side3 = self.sidelayer3(x)
        #     x_side3 = self.bn_one(x_side3[0])
        #     x_side3 = self.sigmoid(x_side3)
        #     size_x = x_side2_concat.size()
        #     x_side2_concat = x_side2_concat.chunk(size_x[2], 2)
        #     x_side3 = x_side3.chunk(size_x[2], 2)
        #     x_side3_concat = torch.cat((x_side2_concat[0], x_side3[0], x_side2_concat[1], x_side3[1],
        #                                 x_side2_concat[2], x_side3[2], x_side2_concat[3], x_side3[3]), 2)
        #     x_side3_concat = self.average_pooling(x_side3_concat)
        #
        # x = self.s5(x)
        #
        # if self.cfg.MODEL.SSF_ENABLE:
        #     F = self.fusion_temp_mask(x_side3_concat)
        #     x = torch.mul(F, x[0])
        #     x = self.head(x)
        # else:
        #     x = self.head(x[0])
        #
        # x = x.view(-1, self.fc.in_features)
        # x = self.fc(self.dropout(x))
        #
        # if self.cfg.MODEL.LOSS_FUNC == "CE_L1_L2":
        #     return [x, F, x_side1, x_side1_loss, x_side2_concat, x_side2_concat_loss]
        # elif self.cfg.MODEL.LOSS_FUNC == "CE_L1":
        #     return [x, F]
        # elif self.cfg.MODEL.LOSS_FUNC == "cross_entropy":
        #     return [x]
        #
