from __future__ import division
from __future__ import print_function


from . import resnet, video_model_builder

def generate_model(cfg):

    if cfg.MODEL.MODEL_NAME == 'resnet10':
        model = resnet.resnet10(num_classes=cfg.MODEL.NUM_CLASSES, shortcut_type=cfg.MODEL.RESNET_SHORTCUT)

    elif cfg.MODEL.MODEL_NAME == 'resnet18':
        model = resnet.resnet18(num_classes=cfg.MODEL.NUM_CLASSES, shortcut_type=cfg.MODEL.RESNET_SHORTCUT)

    elif cfg.MODEL.MODEL_NAME == 'resnet34':
        model = resnet.resnet34(num_classes=cfg.MODEL.NUM_CLASSES, shortcut_type=cfg.MODEL.RESNET_SHORTCUT)

    elif cfg.MODEL.MODEL_NAME == 'resnet50':
        model = resnet.resnet50(num_classes=cfg.MODEL.NUM_CLASSES, shortcut_type=cfg.MODEL.RESNET_SHORTCUT)

    elif cfg.MODEL.MODEL_NAME == 'resnet101':
        model = resnet.resnet101(num_classes=cfg.MODEL.NUM_CLASSES, shortcut_type=cfg.MODEL.RESNET_SHORTCUT)

    elif cfg.MODEL.MODEL_NAME == 'resnet152':
        model = resnet.resnet152(num_classes=cfg.MODEL.NUM_CLASSES, shortcut_type=cfg.MODEL.RESNET_SHORTCUT)

    elif cfg.MODEL.MODEL_NAME == 'resnet200':
        model = resnet.resnet200(num_classes=cfg.MODEL.NUM_CLASSES, shortcut_type=cfg.MODEL.RESNET_SHORTCUT)

    elif cfg.MODEL.MODEL_NAME == 'i3d_non_local_res50':
        model = video_model_builder.ResNet(cfg)

    return model
#
