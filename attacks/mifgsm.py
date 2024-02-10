from torch.autograd import grad

from attacks.base import *


class MIFGSMAttack(BaseAttack):
    
  def __init__(self, model:Model, dfn:Callable=None, eps:float=0.03, alpha:float=0.01, steps:int=40, decay:float=1.0, random_start:bool=True, **kwargs):
    super().__init__(model, dfn)

    self.eps = eps
    self.alpha = alpha
    self.steps = steps
    self.decay = decay
    self.random_start = random_start
        
  @torch.no_grad()
  def __call__(self, X:Tensor, Y:Tensor) -> Tensor:
    X = X.clone().detach()
    Y = Y.clone().detach()
    
    AX = X.clone().detach()
    
    if self.random_start:
      AX = AX + torch.empty_like(AX).uniform_(-self.eps, self.eps)
      AX = self.std_clip(AX).detach()
      
    momentum = torch.zeros_like(X).detach()
    
    self.model.eval()
    for _ in tqdm(range(self.steps)):
      AX.requires_grad = True
      
      with torch.enable_grad():
        outputs = self.model_forward(AX)
        loss = F.cross_entropy(outputs, Y)
      
      g = grad(loss, AX, grad_outputs=loss)[0]
      g = g / torch.mean(torch.abs(g), dim=(1, 2, 3), keepdim=True)
      g = g + momentum * self.decay
      momentum = g
      
      AX = AX.detach() + self.alpha * g.sign()
      delta = torch.clamp(AX - X, min=-self.eps, max=self.eps)
      AX = self.std_clip(X + delta).detach()
    
    return self.std_quant(AX)


if __name__ == '__main__':
  from attacks.unit_test import unittest
  
  unittest(MIFGSMAttack)
