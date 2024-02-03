#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/03

import sys
from time import time
from datetime import datetime
from pprint import pprint as pp

from data import ImageNet_1k, NIPS17_pair, normalize, DataLoader
from model import get_model, MODELS
from attacks import PGDAttack
from defenses import PatchReplaceDefense
from utils import *


@torch.no_grad()
def run_metrics(model:Model, dataloader:DataLoader, atk:Callable=IDENTITY, dfn:Callable=IDENTITY) -> tuple:
  total, correct = 0, 0
  rcorrect, changed, twisted = 0, 0, 0

  model.eval()
  for X, Y in tqdm(dataloader):
    X, Y = X.to(device), Y.to(device)
    AX = atk(X, Y)

    with torch.inference_mode():
      pred_X : Tensor = model(normalize(dfn(X ))).argmax(dim=-1)
      pred_AX: Tensor = model(normalize(dfn(AX))).argmax(dim=-1) if atk is not IDENTITY else pred_X

    total    += len(pred_X)
    correct  += (pred_X  == Y)                  .sum().item()
    rcorrect += (pred_AX == Y)                  .sum().item()
    changed  += (pred_AX != pred_X)             .sum().item()
    twisted  += ((pred_X == Y) & (pred_AX != Y)).sum().item()

  safe_div = lambda x, y: (x / y) if y else 0
  return [
    safe_div(correct,  total),    # Clean Accuracy: clean correct
    safe_div(rcorrect, total),    # Remnant Accuracy: adversarial still correct
    safe_div(changed,  total),    # Prediction Change Rate: prediction changed under attack
    safe_div(twisted,  correct),  # Attack Twist Rate: clean correct but adversarial wrong
  ]


def run(args):
  db = db_load(args.logdb)
  try:
    seed_everything(args.seed)
    model = get_model(args.model).to(device)
    dataset = ImageNet_1k()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)
    dfn = PatchReplaceDefense() if args.dfn else IDENTITY
    atk = PGDAttack(model, args.eps, args.alpha, args.steps, not args.nrs, dfn) if args.atk else IDENTITY

    t = time()
    acc, racc, pcr, atr = run_metrics(model, dataloader, atk, dfn)
    ts = time() - t
    rec = {
      'cmd': ' '.join(sys.argv),
      'args': vars(args),
      'dt': str(datetime.now()),
      'ts': ts,
      'metrics': {
        'acc': acc,
        'racc': racc,
        'asr': 1 - racc,
        'pcr': pcr,
        'atr': atr,
      },
    }
    print(f'[{args.model}]:')
    pp(rec, sort_dicts=False)

    db_add(db, args.model, rec)
  except:
    print_exc()
  finally:
    db_save(db, args.logdb)


if __name__ == '__main__':
  parser = ArgumentParser()
  # model & data
  parser.add_argument('-M', '--model', default='resnet18', choices=MODELS, help='model name')
  parser.add_argument('-B', '--batch_size', type=int, default=20, help='run batch size')
  # attack
  parser.add_argument('--atk', action='store_true')
  parser.add_argument('--eps',   type=eval, default=8/255, help='PGD total threshold')
  parser.add_argument('--alpha', type=eval, default=1/255, help='PGD step size')
  parser.add_argument('--steps', type=int,  default=10,    help='PGD step count')
  parser.add_argument('--nrs', action='store_true', help='no random start, like the original FGSM')
  # defense
  parser.add_argument('--dfn', action='store_true')
  # misc
  parser.add_argument('--seed', default=114514, type=int, help='randseed')
  parser.add_argument('--logdb', default=LOG_PATH / 'run.json', type=Path, help='log records file')
  args = parser.parse_args()

  run(args)
