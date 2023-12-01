#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/27 

from pathlib import Path

from pathlib import Path
from PIL import Image
from PIL.Image import Image as PILImage

import torch
import numpy as np
from numpy import ndarray

BASE_PATH = Path(__file__).parent
DATA_PATH = BASE_PATH / 'data'
DATA_RAW_PATH = DATA_PATH / 'NIPS17'
DATA_ADV_PATH = DATA_PATH / 'ssa-cwa-200'

npimg = ndarray

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cpu = 'cpu'   # for qt model

if device == 'cuda':
  torch.backends.cudnn.enabled = False
  torch.backends.cudnn.benchmark = False


def load_img(fp:Path) -> PILImage:
  return Image.open(fp).convert('RGB')

def pil_to_npimg(img:PILImage) -> npimg:
  return np.asarray(img, dtype=np.uint8)

def npimg_to_pil(im:npimg) -> PILImage:
  assert im.dtype in [np.uint8, np.float32]
  if im.dtype == np.float32:
    assert 0.0 <= im.min() and im.max() <= 1.0
  return Image.fromarray(im)

def to_ch_avg(x:ndarray) -> ndarray:
  return np.tile(x.mean(axis=-1, keepdims=True), (1, 1, 3))

def to_gray(im:npimg) -> npimg:
  return pil_to_npimg(npimg_to_pil(im).convert('L'))

def minmax_norm(dx:ndarray, vmin:int=None, vmax:int=None) -> npimg:
  vmin = vmin or dx.min()
  vmax = vmax or dx.max()
  out = (dx - vmin) / (vmax - vmin)
  return (out * 255).astype(np.uint8)

def npimg_abs_diff(x:npimg, y:npimg, name:str=None) -> npimg:
  d = np.abs(x.astype(np.int16) - y.astype(np.int16))
  if name:
    print(f'[{name}]')
    print('  Linf:', d.max() / 255)
    print('  L1:',  d.mean() / 255)
  return d
