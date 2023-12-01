#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/12/01

# test the pre-generated adv images on a different model & task

from argparse import ArgumentParser
from traceback import print_exc

import torch
from tqdm import tqdm

from model import get_model, MODELS
from data import NIPS17_pair, DataLoader
from utils import device


@torch.inference_mode()
def do_test(model, dataloader) -> float:
  total, match = 0, 0

  model.eval()
  for X, AX in tqdm(dataloader):
    X  = X .to(device)
    AX = AX.to(device)

    pred_X  = model(X) .argmax(dim=-1)
    pred_AX = model(AX).argmax(dim=-1)

    total += len(pred_X)
    match += (pred_X == pred_AX).sum().item()

  return match / total


def test(args):
  ''' Model '''
  model = get_model(args.model).to(device)

  ''' Data '''
  dataset = NIPS17_pair(args.filter)
  dataloader = DataLoader(dataset, args.batch_size, shuffle=args.shuffle)
  
  ''' Test '''
  mr = do_test(model, dataloader)
  print(f'[{args.model}] Match Rate: {mr:.2%}')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-M', '--model', default='resnet18', choices=MODELS)
  parser.add_argument('-B', '--batch_size', type=int, default=64)
  parser.add_argument('--filter', default='none', choices=['none', 'low', 'high'])
  parser.add_argument('--shuffle', action='store_true')
  parser.add_argument('--run_all', action='store_true')
  args = parser.parse_args()

  if args.run_all:
    for model in MODELS:
      args.model = model
      try:
        test(args)
      except:
        print('>> model {model} failed')
        print_exc()
  else:
    test(args)
