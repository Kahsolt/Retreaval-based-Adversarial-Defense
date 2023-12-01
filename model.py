#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/30 

import warnings ; warnings.simplefilter('ignore')

from torch.nn import Module
import torchvision.models as M

MODELS = [
  'resnet18',
  'resnet34',
  'resnet50',
  'resnet101',
  'resnet152',
  'resnext50_32x4d',
  'resnext101_32x8d',
  'resnext101_64x4d',
  'wide_resnet50_2',
  'wide_resnet101_2',

  'convnext_tiny',
  'convnext_small',
  'convnext_base',
  'convnext_large',

  'densenet121',
  'densenet161',
  'densenet169',
  'densenet201',

  'vit_b_16',
  'vit_b_32',
  'vit_l_16',
  'vit_l_32',
  'vit_h_14',

  'swin_t',
  'swin_s',
  'swin_b',
  'swin_v2_t',
  'swin_v2_s',
  'swin_v2_b',

  'maxvit_t',

  'inception_v3',

  'squeezenet1_0',
  'squeezenet1_1',

  'mobilenet_v2',
  'mobilenet_v3_small',
  'mobilenet_v3_large',

  'shufflenet_v2_x0_5',
  'shufflenet_v2_x1_0',
  'shufflenet_v2_x1_5',
  'shufflenet_v2_x2_0',
]


def get_model(name) -> Module:
  return getattr(M, name)(pretrained=True)
