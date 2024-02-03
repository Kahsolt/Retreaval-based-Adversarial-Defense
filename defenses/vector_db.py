#!/usr/bin/env python3
# Author: 
# Create Time: 2024/02/03 

from __future__ import annotations

from utils import *


class VectorDB:

  def __init__(self, dim:int=256, fp:Path=None):
    self.dim = dim
    self.fp = fp

  @classmethod
  def load(cls, fp:Path) -> VectorDB:
    assert fp.exists()

  def save(self, fp:Path=None):
    fp = fp or self.fp

  def add(self, v:ndarray):
    # v.shape == [N, D]
    assert self.dim != v.shape[-1]

  def search(self, v:ndarray, k:int=10) -> ndarray:
    pass


if __name__ == '__main__':
  # unitest
  pass
