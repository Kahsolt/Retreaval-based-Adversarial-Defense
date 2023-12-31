#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/12/29 

import warnings ; warnings.filterwarnings('ignore', category=UserWarning)

import os
import sys
import json
import random
from PIL import Image
from pathlib import Path
from time import time
from datetime import datetime
from argparse import ArgumentParser, Namespace
from traceback import print_exc
from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.autograd import grad
from torch.utils.data import DataLoader
import torchvision.models as M
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
from tqdm import tqdm

from defenses import BlurAndSharpen

BASE_PATH = Path(__file__).parent.absolute()
DATA_PATH = BASE_PATH / 'data'
DB_FILE = BASE_PATH / 'run.json'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def seed_everything(seed:int) -> int:
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

  if device == 'cuda':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


Model = nn.Module
Record = Dict[str, Any]
DB = Dict[str, List[Record]]

'''
{
  'model': [{
    ts: int
    cmd: str
    args: dict
    acc: float
    racc: float
    pcr: float
    asr: float
  }]
}
'''

def db_load() -> DB:
  if not DB_FILE.exists():
    return {}
  else:
    with open(DB_FILE, 'r', encoding='utf-8') as fh:
      return json.load(fh)

def db_save(db:DB):
  with open(DB_FILE, 'w', encoding='utf-8') as fh:
    json.dump(db, fh, indent=2, ensure_ascii=False)

def db_add(db:DB, model:str, rec:Record):
  if model in db:
    db[model].append(rec)
  else:
    db[model] = [rec]

MODELS = [
  'resnet18',
  'resnet50',
  'wide_resnet50_2',
  'resnext50_32x4d',
  'densenet121',
  'regnet_x_8gf',
  'regnet_y_8gf',
  'convnext_base',
  'mnasnet0_75',
  'mnasnet1_3',
  'mobilenet_v2',
  'mobilenet_v3_large',
  'squeezenet1_1',
  'shufflenet_v2_x2_0',
  'efficientnet_b0',
  'efficientnet_v2_m',
  'swin_b',
  'swin_v2_b',
  'vit_b_16',
  'vit_b_32',
  'maxvit_t',
  'googlenet',
  'inception_v3',
]


class ImageNet_1k(Dataset):

  def __init__(self, root:str, limit:int=-1, shuffle:bool=False):
    self.base_path = os.path.join(root, 'val')

    fns = [fn for fn in os.listdir(self.base_path)]
    fps = [os.path.join(self.base_path, fn) for fn in fns]
    with open(os.path.join(root, 'image_name_to_class_id_and_name.json'), encoding='utf-8') as fh:
      mapping = json.load(fh)
    tgts = [mapping[fn]['class_id'] for fn in fns]

    self.metadata = [x for x in zip(fps, tgts)]
    if shuffle: random.shuffle(self.metadata)
    if limit > 0: self.metadata = self.metadata[:limit]

  def __len__(self):
    return len(self.metadata)

  def __getitem__(self, idx):
    fp, tgt = self.metadata[idx]
    img = Image.open(fp).convert('RGB')

    if 'use numpy':
      im = np.asarray(img, dtype=np.uint8).transpose(2, 0, 1)   # [C, H, W]
      im = im / np.float32(255.0)
    else:
      im = T.ToTensor()(img)

    return im, tgt


def normalize(X: torch.Tensor) -> torch.Tensor:
  ''' NOTE: to insure attack validity, normalization is delayed until put into model '''

  mean = (0.485, 0.456, 0.406)
  std  = (0.229, 0.224, 0.225)
  return TF.normalize(X, mean, std)       # [B, C, H, W]


def get_dataloader(batch_size=32, limit=-1, shuffle=False):
  dataset = ImageNet_1k(DATA_PATH / 'imagenet-1k', limit, shuffle)
  dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=0)
  return dataloader


def get_model(name:str) -> Model:
  model: Model = getattr(M,  name)(pretrained=True)
  return model.eval().to(device)


def pgd(model:Model, dfn:BlurAndSharpen, X:Tensor, Y:Tensor, eps:float=0.03, alpha:float=0.001, steps:int=40, random_start:bool=True) -> Tensor:
  X = X.clone().detach()
  Y = Y.clone().detach()

  AX = X.clone().detach()
  if random_start:
    AX = AX + torch.empty_like(AX).uniform_(-eps, eps)
    AX = torch.clamp(AX, min=0.0, max=1.0).detach()

  for _ in tqdm(range(steps)):
    AX.requires_grad = True

    with torch.enable_grad():
      logits = model(normalize(dfn(AX)))
      loss = F.cross_entropy(logits, Y, reduction='none')

    g = grad(loss, AX, grad_outputs=loss)[0]
    AX = AX.detach() + alpha * g.sign()
    delta = torch.clamp(AX - X, min=-eps, max=eps)
    AX = torch.clamp(X + delta, min=0.0, max=1.0).detach()

  # assure valid rgb pixel
  return (AX * 255).round().div(255.0)


@torch.no_grad()
def test_atk(args:Namespace, model:Model, dfn:BlurAndSharpen, dataloader:DataLoader) -> tuple:
  total, correct = 0, 0
  rcorrect, changed, twisted = 0, 0, 0

  model.eval()
  for X, Y in tqdm(dataloader):
    X, Y = X.to(device), Y.to(device)

    AX = pgd(model, dfn, X, Y, args.eps, args.alpha, args.step, not args.nrs) if args.atk else X

    with torch.inference_mode():
      pred   : Tensor = model(normalize(dfn(X ))).argmax(dim=-1)
      pred_AX: Tensor = model(normalize(dfn(AX))).argmax(dim=-1) if args.atk else pred

    total    += len(pred)
    correct  += (pred    == Y   )             .sum().item()
    rcorrect += (pred_AX == Y   )             .sum().item()
    changed  += (pred_AX != pred)             .sum().item()
    twisted  += ((pred == Y) & (pred_AX != Y)).sum().item()

  safe_div = lambda x, y: (x / y) if y else 0
  return [
    safe_div(correct,  total),    # Clean Accuracy: clean correct
    safe_div(rcorrect, total),    # Remnant Accuracy: adversarial still correct
    safe_div(changed,  total),    # Prediction Change Rate: prediction changed under attack
    safe_div(twisted,  correct),  # Attack Twist Rate: clean correct but adversarial wrong
  ]


def run(args):
  models = MODELS if args.run_all else [args.model]
  dataloader = get_dataloader(args.batch_size)
  dfn = BlurAndSharpen(k=args.k, s=args.s) if args.dfn else nn.Identity()

  db = db_load()
  try:
    for name in models:
      args.model = name
      try:
        seed_everything(args.seed)
        model = get_model(args.model)

        t = time()
        acc, racc, pcr, atr = test_atk(args, model, dfn, dataloader)
        ts = time() - t
        rec = {
          'dt': str(datetime.now()),
          'cmd': ' '.join(sys.argv),
          'ts': ts,
          'args': vars(args),
          'metrics': {
            'acc': acc,
            'racc': racc,
            'asr': 1 - racc,
            'pcr': pcr,
            'atr': atr,
          },
        }
        print(f'[{name}]:', rec)

        db_add(db, name, rec)
      except:
        print_exc()
        print(f'[{name}] failed')
  finally:
    db_save(db)


def get_args():
  parser = ArgumentParser()
  # model & data
  parser.add_argument('-M', '--model', default='resnet18', choices=MODELS, help='model name')
  parser.add_argument('-B', '--batch_size', type=int, default=20, help='run batch size')
  # attack
  parser.add_argument('--atk', action='store_true')
  parser.add_argument('--eps',   type=eval, default=8/255, help='PGD total threshold')
  parser.add_argument('--alpha', type=eval, default=1/255, help='PGD step size')
  parser.add_argument('--step',  type=int,  default=10,    help='PGD step count')
  parser.add_argument('-nrs', action='store_true', help='no random start, like the original FGSM')
  # defense
  parser.add_argument('--dfn', action='store_true')
  parser.add_argument('--k', type=int,  default=5,   help='BlurAndSharpen blur kernel size')
  parser.add_argument('--s', type=eval, default=1.5, help='BlurAndSharpen sharpen factor: blurred(<1), original(=1), sharpened(>1)')
  # misc
  parser.add_argument('--seed', default=114514, type=int, help='randseed')
  parser.add_argument('--run_all', action='store_true')
  return parser.parse_args()


if __name__ == '__main__':
  args = get_args()
  if args.k > 0:
    assert args.k % 2 == 1 and args.k > 1, '--k must be odd integer > 1'
  run(args)
