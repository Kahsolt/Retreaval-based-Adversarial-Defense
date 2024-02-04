from torch.autograd import grad

from data import normalize
from utils import *

class FGSMAttack:
  
  def __init__(self, model:Model, eps:float=0.03, dfn:Callable=None):
    self.model = model
    self.dfn = dfn or (lambda _: _)
    self.eps = eps
    
  def __call__(self, X:Tensor, Y:Tensor) -> Tensor:
    X = X.clone().detach()
    Y = Y.clone().detach()
    self.model.eval()
    
    with torch.enable_grad():
      AX.requires_grad = True
      logits = self.model(normalize(self.dfn(AX)))
      loss = F.cross_entropy(logits, Y, reduction='none')
    
    g = grad(loss, AX, grad_outputs=loss)[0]
    AX = X + self.eps * g.sign()
    return (AX * 255).round().div(255.0)
  
if __name__ == '__main__':
  from plot import *

  # unitest
  pass
    