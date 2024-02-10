#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/12/29 

from torch.autograd import grad

from attacks.base import *


class PGDAttack(BaseAttack):

  def __init__(self, model:Model, dfn:Callable=None, eps:float=8/255, alpha:float=1/255, steps:int=20, random_start:bool=True, **kwargs):
    super().__init__(model, dfn)

    self.eps = eps
    self.alpha = alpha
    self.steps = steps
    self.random_start = random_start

  @torch.no_grad()
  def __call__(self, X:Tensor, Y:Tensor) -> Tensor:
    X = X.clone().detach()
    Y = Y.clone().detach()

    AX = X.clone().detach()
    if self.random_start:
      AX = AX + torch.empty_like(AX).uniform_(-self.eps, self.eps)
      AX = self.std_clip(AX).detach()

    self.model.eval()
    for _ in tqdm(range(self.steps)):
      AX.requires_grad = True

      with torch.enable_grad():
        logits = self.model_forward(AX)
        loss = F.cross_entropy(logits, Y, reduction='none')

      g = grad(loss, AX, grad_outputs=loss)[0]
      AX = AX.detach() + self.alpha * g.sign()
      DX = torch.clamp(AX - X, min=-self.eps, max=self.eps)
      AX = self.std_clip(X + DX).detach()

    return self.std_quant(AX)


if __name__ == '__main__':
  from attacks.unit_test import unittest
  
  unittest(PGDAttack)
