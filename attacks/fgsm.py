from torch.autograd import grad

from attacks.base import *


class FGSMAttack(BaseAttack):
  
  def __init__(self, model:Model, dfn:Callable=None, eps:float=0.03, **kwargs):
    super().__init__(model, dfn)
    
    self.eps = eps
  
  @torch.no_grad()
  def __call__(self, X:Tensor, Y:Tensor) -> Tensor:
    X = X.clone().detach()
    Y = Y.clone().detach()
    
    AX = X.clone().detach()
    self.model.eval()
    with torch.enable_grad():
      AX.requires_grad = True
      logits = self.model_forward(AX)
      loss = F.cross_entropy(logits, Y, reduction='none')

    g = grad(loss, AX, grad_outputs=loss)[0]
    AX = X + self.eps * g.sign()
    AX = self.std_clip(AX).detach()
    
    return self.std_quant(AX)
  
if __name__ == '__main__':
  from unit_test import unittest

  unittest(FGSMAttack)
  # unitest
  pass
    