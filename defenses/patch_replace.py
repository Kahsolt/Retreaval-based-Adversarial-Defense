#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/03

from .vector_db import VectorDB

from utils import *


class PatchReplaceDefense:
  
  def __init__(self):
    self.vdb = VectorDB()

  def __call__(self, X:Tensor) -> Tensor:
    return X


if __name__ == '__main__':
  from plot import *

  # unitest
  pass
