#!/usr/bin/env python3
# Author: Shoi
# Create Time: 2024/02/03 

from __future__ import annotations

from utils import *
import faiss

class VectorDB:

  def __init__(self, index:faiss.IndexFlatL2=None, dim:int=256, fp:Path=None):
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
    # fp = "xxx.index"
    fp = fp or self.fp
    assert fp != None, "no Path"
    assert fp.suffix == ".index","xxx.index"

    faiss.write_index(self.index, str(fp))

  def add(self, v:ndarray):
    # v.shape == [N, D]
    assert self.dim == v.shape[-1],"dim error"
    self.index.add(v)


  def search(self, v:ndarray, k:int=10) -> (ndarray, ndarray):
    assert self.dim == v.shape[-1],"dim error"
    # list and distance
    return self.index.search(v, k)


if __name__ == '__main__':
  # data
  dim = 64
  data = np.random.rand(500, dim).astype('float32')
  search_data = np.random.random((5, dim)).astype('float32')

  # init
  index = faiss.IndexFlatL2(dim) 
  DB = VectorDB(index, dim)

  # load data
  # print(Path.cwd()/'defenses/index.index')
  # DB = VectorDB.load(Path.cwd()/'defenses/index.index')

  # add 
  DB.add(data)
  print(DB.index.ntotal)

  # search
  D,I = DB.search(search_data, 4)
  print(I)
  print(D)

  # save
  # DB.save(Path.cwd()/'defenses/newindex.index')
  DB.save()

  # unitest
  pass
