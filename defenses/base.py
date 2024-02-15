#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/2/15

from utils import *


class BaseDefense:

  @torch.no_grad()
  def __call__(self, X:Tensor) -> Tensor:
    raise NotImplementedError
