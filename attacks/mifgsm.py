from torch.autograd import grad

from data import normalize
from utils import *

class MIFGSMAttack:
    
  def __init__(self, model:Model, eps:float=0.03, alpha:float=0.01, steps:int=40, random_start:bool=True, dfn:Callable=None):
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
    print('X.shape:', X.shape)
    self.model.eval()
    momentum = torch.zeros_like(X).detach()
    loss = torch.nn.CrossEntropyLoss()
    
    for _ in range(self.steps):
      AX.requires_grad = True
      outputs = self.model(normalize(self.dfn(AX)))
      cost = loss(outputs, Y)
      grad = torch.autograd.grad(cost, AX, retain_graph=False, create_graph=False)[0]
      grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
      grad = grad + momentum * 1.0
      momentum = grad
      AX = AX.detach() + self.alpha * grad.sign()
      delta = torch.clamp(AX - X, min=-self.eps, max=self.eps)
      AX = torch.clamp(X + delta, min=0, max=1).detach()
    
    return (AX * 255).round().div(255.0)
  
if __name__ == '__main__':
  from plot import *
  # unitest
  pass