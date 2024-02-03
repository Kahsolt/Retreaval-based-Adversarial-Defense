#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/12/29 

from torch.autograd import grad

from data import normalize
from utils import *


class PGDAttack:

  def __init__(self, model:Model, eps:float=0.03, alpha:float=0.001, steps:int=40, random_start:bool=True, dfn:Callable=None):
    self.model = model
    self.dfn = dfn or (lambda _: _)
    self.eps = eps
    self.alpha = alpha
    self.steps = steps
    self.random_start = random_start

  def __call__(self, X:Tensor, Y:Tensor) -> Tensor:
    X = X.clone().detach()
    Y = Y.clone().detach()

    AX = X.clone().detach()
    if self.random_start:
      AX = AX + torch.empty_like(AX).uniform_(-self.eps, self.eps)
      AX = torch.clamp(AX, min=0.0, max=1.0).detach()

    self.model.eval()
    for _ in tqdm(range(self.steps)):
      AX.requires_grad = True

      with torch.enable_grad():
        logits = self.model(normalize(self.dfn(AX)))
        loss = F.cross_entropy(logits, Y, reduction='none')

      g = grad(loss, AX, grad_outputs=loss)[0]
      AX = AX.detach() + self.alpha * g.sign()
      DX = torch.clamp(AX - X, min=-self.eps, max=self.eps)
      AX = torch.clamp(X + DX, min=0.0, max=1.0).detach()

    return (AX * 255).round().div(255.0)


if __name__ == '__main__':
  from plot import *

  # unitest
  pass
