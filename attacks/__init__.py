from .pgd import PGDAttack
from .fgsm import FGSMAttack
from .mifgsm import MIFGSMAttack


def get_attack(name:str, args, model, dfn):
  if name == 'PGD':
    return PGDAttack(model, args.eps, args.alpha, args.steps, not args.nrs, dfn)
  elif name == 'FGSM':
    return FGSMAttack(model, args.eps, dfn)
  elif name == 'MIFGSM':
    return MIFGSMAttack(model, args.eps, args.alpha, args.steps, not args.nrs, dfn)
