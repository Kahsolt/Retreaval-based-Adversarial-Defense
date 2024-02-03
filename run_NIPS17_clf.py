#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/12/01

# run the NIPS17 pre-generated adv images on various clf-models

from data import NIPS17_pair, normalize, DataLoader
from model import get_model, MODELS
from utils import *


@torch.inference_mode()
def run_metrics(model:Model, dataloader:DataLoader) -> float:
  total, match = 0, 0

  model.eval()
  for X, AX in tqdm(dataloader):
    X  = X .to(device)
    AX = AX.to(device)

    pred_X : Tensor = model(normalize(X)) .argmax(dim=-1)
    pred_AX: Tensor = model(normalize(AX)).argmax(dim=-1)

    total += len(pred_X)
    match += (pred_X == pred_AX).sum().item()

  return match / total


def run(args):
  seed_everything(args.seed)
  model = get_model(args.model).to(device)
  dataset = NIPS17_pair(args.filter)
  dataloader = DataLoader(dataset, args.batch_size, shuffle=args.shuffle)

  mr = run_metrics(model, dataloader)
  print(f'[{args.model}] Match Rate: {mr:.2%}')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-M', '--model', default='resnet18', choices=MODELS)
  parser.add_argument('-B', '--batch_size', type=int, default=64)
  parser.add_argument('--filter', default='none', choices=['none', 'low', 'high'])
  parser.add_argument('--shuffle', action='store_true')
  parser.add_argument('--run_all', action='store_true')
  parser.add_argument('--seed', default=114514, type=int, help='randseed')
  args = parser.parse_args()

  if args.run_all:
    for model in MODELS:
      args.model = model
      try:
        run(args)
      except KeyboardInterrupt:
        print('>> Exit by Ctrl+C')
        exit(0)
      except:
        print(f'>> model {model} failed')
        print_exc()
  else:
    run(args)
