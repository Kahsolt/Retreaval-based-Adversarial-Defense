#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/27 

import warnings ; warnings.filterwarnings('ignore', category=UserWarning)

import json
import random
from time import time
from pathlib import Path
from PIL import Image, ImageFilter
from PIL.Image import Image as PILImage
from argparse import ArgumentParser, Namespace
from traceback import print_exc
from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module as Model
import numpy as np
from numpy import ndarray
from numpy.typing import NDArray
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

IDENTITY = lambda *args: args[0] if len(args) == 1 else args

BASE_PATH = Path(__file__).parent
LOG_PATH = BASE_PATH / 'log' ; LOG_PATH.mkdir(exist_ok=True)
DATA_PATH = BASE_PATH / 'data'
DATA_IMAGENET_1K_PATH = DATA_PATH / 'imagenet-1k'
DATA_NIPS17_RAW_PATH = DATA_PATH / 'NIPS17'
DATA_NIPS17_ADV_PATH = DATA_PATH / 'ssa-cwa-200'

IM_U8_TYPES = ['u', 'u8', 'uint8', np.uint8]
IM_F32_TYPES = ['f', 'f32', 'float32', np.float32]
IM_TYPES = IM_U8_TYPES + IM_F32_TYPES

npimg_u8 = NDArray[np.uint8]        # vrng [0, 255]
npimg_f32 = NDArray[np.float32]     # vrng [0, 1]
npimg = Union[npimg_u8, npimg_f32]
npimg_dx = NDArray[np.int16]        # vrng [-255, 255]
npimg_hi = NDArray[np.float32]      # vrng [-1, 1]


def timer(fn):
  def wrapper(*args, **kwargs):
    start = time()
    r = fn(*args, **kwargs)
    end = time()
    print(f'[Timer]: {fn.__name__} took {end - start:.3f}s')
    return r
  return wrapper

def seed_everything(seed:int) -> int:
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

  if device == 'cuda':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


Record = Dict[str, Any]
DB = Dict[str, List[Record]]

'''
{
  'model': [{
    cmd: str
    args: dict
    dt: str
    ts: int
    acc: float
    racc: float
    pcr: float
    asr: float
  }]
}
'''

def db_load(fp:Path) -> DB:
  if not fp.exists(): return {}
  with open(fp, 'r', encoding='utf-8') as fh:
    return json.load(fh)

def db_save(db:DB, fp:Path):
  def cvt(v:Any) -> str:
    if isinstance(v, Path): return str(v)
    return v
  with open(fp, 'w', encoding='utf-8') as fh:
    json.dump(db, fh, indent=2, ensure_ascii=False, default=cvt)

def db_add(db:DB, model:str, rec:Record):
  if model in db: db[model].append(rec)
  else:           db[model] = [rec]


def load_img(fp:Path) -> PILImage:
  return Image.open(fp).convert('RGB')

def load_im(fp:Path, dtype:str='u8') -> npimg:
  return pil_to_npimg(load_img(fp), dtype)

def pil_to_npimg(img:PILImage, dtype:str='u8') -> npimg:
  assert dtype in IM_TYPES
  im = np.asarray(img, dtype=np.uint8)
  if dtype in IM_U8_TYPES: return im
  return im.astype(np.float32) / 255.0

def npimg_to_pil(im:npimg) -> PILImage:
  assert im.dtype in IM_TYPES
  if im.dtype in IM_F32_TYPES:
    assert 0.0 <= im.min() and im.max() <= 1.0
    im = (im * 255.0).astype(np.uint8)
  return Image.fromarray(im)

def hwc2chw(im:npimg) -> npimg:
  return im.transpose(2, 0, 1)

def chw2hwc(im:npimg) -> npimg:
  return im.transpose(1, 2, 0)

def npimg_to_tensor(im:npimg_f32) -> Tensor:
  return torch.from_numpy(hwc2chw(im))

def tensor_to_npimg(X:Tensor) -> npimg_f32:
  return chw2hwc(X.detach().cpu().numpy())

def std_clip(im:npimg_f32) -> npimg_f32:
  return np.clip(im, 0.0, 1.0)

def to_gray(im:npimg) -> npimg:
  return pil_to_npimg(npimg_to_pil(im).convert('L'))

def to_ch_avg(x:ndarray) -> ndarray:
  return np.tile(x.mean(axis=-1, keepdims=True), (1, 1, 3))

def npimg_abs_diff(x:npimg, y:npimg, name:str=None) -> npimg:
  d: ndarray = np.abs(npimg_diff(x, y))
  if name:
    print(f'[{name}]')
    print('  Linf:', d.max() / 255)
    print('  L1:',  d.mean() / 255)
  return d

def npimg_diff(x:npimg_u8, y:npimg_u8) -> npimg_dx:
  return x.astype(np.int16) - y.astype(np.int16)

def minmax_norm(dx:npimg_dx, vmin:int=None, vmax:int=None) -> npimg_u8:
  vmin = vmin or dx.min()
  vmax = vmax or dx.max()
  out = (dx - vmin) / (vmax - vmin)
  return (out * 255).astype(np.uint8)

def Linf_L1_L2(X:Tensor, AX:Tensor=None) -> Tuple[float, float, float]:
  if AX is None:
    DX = X
  else:
    DX = (AX - X).abs()
  Linf = DX.max()
  L1 = DX.mean()
  L2 = (DX**2).sum().sqrt()
  return [x.item() for x in [Linf, L1, L2]]
