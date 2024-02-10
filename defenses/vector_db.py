#!/usr/bin/env python3
# Author: Shoi
# Create Time: 2024/02/03 

from __future__ import annotations

import faiss
from faiss import IndexFlatL2

from utils import *


class VectorDB:

  def __init__(self, index:IndexFlatL2=None, dim:int=256, fp:Path=None):
    self.index = index
    self.dim = dim
    self.fp = fp

  @classmethod
  def load(cls, fp:Path) -> VectorDB:
    assert fp.exists()
    index = faiss.read_index(str(fp))
    dim = index.d
    return VectorDB(index, dim, fp)

  def save(self, fp:Path=None):
    fp = fp or self.fp
    assert fp != None, "no Path"
    assert fp.suffix == ".index", "xxx.index"

    faiss.write_index(self.index, str(fp))

  def add(self, v:ndarray):
    # v.shape == [N, D]
    assert self.dim == v.shape[-1], "dim error"
    self.index.add(v)

  def search(self, v:ndarray, k:int=10) -> Tuple[ndarray, ndarray]:
    assert self.dim == v.shape[-1], "dim error"
    # list and distance
    return self.index.search(v, k)


if __name__ == '__main__':
  from utils import LOG_PATH
  index_fp = LOG_PATH / 'test.index'

  np.random.seed(114514)

  # load data
  if index_fp.exists():
    print('>> load from index file:', index_fp)
    vec_db = VectorDB.load(index_fp)
    dim = vec_db.dim
  else:
    print('>> make new index file:', index_fp)
    dim = 64
    index = faiss.IndexFlatL2(dim) 
    vec_db = VectorDB(index, dim, index_fp)
    data = np.random.rand(500, dim).astype('float32')
    vec_db.add(data)
  print('dataset size:', vec_db.index.ntotal)

  # search
  search_data = np.random.random((5, dim)).astype('float32')
  D, I = vec_db.search(search_data, 4)
  print('indexes:', I)
  print('dists:', D)

  # save
  vec_db.save()
