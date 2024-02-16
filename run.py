#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/03

import sys
from time import time
from datetime import datetime
from inspect import signature
from pprint import pprint as pp

from data import ImageNet_1k, NIPS17_pair, normalize, DataLoader
from model import get_model, MODELS
from attacks import *
from defenses import *
from utils import *

DEFENSE_METHODS = ['None'] + [k[:-len('Defense')] for k, v in globals().items() if k.endswith('Defense') and issubclass(v, BaseDefense) and v != BaseDefense]
ATTACK_METHODS = ['None'] + [k[:-len('Attack')] for k, v in globals().items() if k.endswith('Attack') and issubclass(v, BaseAttack) and v != BaseAttack]


def get_defense(args) -> BaseAttack:
  if args.dfn == 'None': return IDENTITY
  dfn_cls = globals()[f'{args.dfn}Defense']
  kwargs = { }
  for name in signature(dfn_cls).parameters:
    if hasattr(args, name):
      kwargs[name] = getattr(args, name)
  return dfn_cls(**kwargs)


def get_attack(args, model:Model, dfn:Callable) -> BaseAttack:
  if args.atk == 'None': return IDENTITY
  atk_cls = globals()[f'{args.atk}Attack']
  fixed_args = ['model', 'dfn']
  kwargs = { }
  for name in signature(atk_cls).parameters:
    if name in fixed_args: continue
    if hasattr(args, name):
      kwargs[name] = getattr(args, name)
    else:   # special negatives
      if name == 'random_start':
        kwargs['random_start'] = not args.nrs
  return atk_cls(model, dfn, **kwargs)


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
    dataset = ImageNet_1k(limit=args.limit)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)
    dfn = get_defense(args)
    atk = get_attack(args, model, dfn)

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
  parser.add_argument('-L', '--limit', type=int, default=-1, help='limit run sample count')
  # attack
  parser.add_argument('--atk',   default='None', choices=ATTACK_METHODS, help='attack method')
  parser.add_argument('--eps',   type=eval, default=8/255, help='PGD total threshold')
  parser.add_argument('--alpha', type=eval, default=1/255, help='PGD step size')
  parser.add_argument('--steps', type=int,  default=10,    help='PGD step count')
  parser.add_argument('--nrs', action='store_true', help='no random start, like the original FGSM')
  # defense
  parser.add_argument('--dfn', default='None', choices=DEFENSE_METHODS, help='defense method')
  parser.add_argument('--ref_db',     type=str, default='NIPS17', choices=['NIPS17', 'ImageNet'], help='ref dataset name')
  parser.add_argument('--patch_size', type=int, default=16, help='patch size')
  parser.add_argument('--order',      type=int, default=0,  help='content order of replacement')
  # misc
  parser.add_argument('--seed', default=114514, type=int, help='randseed')
  parser.add_argument('--logdb', default=LOG_PATH / 'run.json', type=Path, help='log records file')
  args = parser.parse_args()

  run(args)
