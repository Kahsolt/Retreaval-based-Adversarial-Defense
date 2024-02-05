#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/2/5 

from data import normalize
from utils import *


class BaseAttack:

  def __init__(self, model:Model, dfn:Callable=None):
    self.model = model
    self.dfn = dfn or IDENTITY

  @property
  def device(self) -> torch.device:
    return next(self.model.parameters())[0].device

  @property
  def dtype(self) -> torch.dtype:
    return next(self.model.parameters())[0].dtype

  def model_forward(self, X:Tensor) -> Tensor:
    return self.model(normalize(self.dfn(X)))

  def std_clip(self, X:Tensor) -> Tensor:
    return torch.clamp(X, min=0.0, max=1.0)

  def std_quant(self, X:Tensor) -> Tensor:
    return (X * 255).round().div(255.0)

  @torch.no_grad()
  def __call__(self, X:Tensor, Y:Tensor=None) -> Tensor:
    raise NotImplementedError
