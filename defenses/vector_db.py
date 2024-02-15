#!/usr/bin/env python3
# Author: Shoi
# Create Time: 2024/02/03 

from __future__ import annotations

import faiss
from faiss import Index, IndexFlatL2

from utils import *


class VectorDB:

  def __init__(self, index_or_dim:Union[int, Index]=256, fp:Path=None):
    assert isinstance(index_or_dim, (int, Index))
    self.index = index_or_dim if isinstance(index_or_dim, Index) else IndexFlatL2(index_or_dim)
    self.fp = fp

  @property
  def dim(self):
    return self.index.d

  @classmethod
  def load(cls, fp:Path) -> VectorDB:
    assert fp.exists()
    index = faiss.read_index(str(fp))
    return VectorDB(index, fp)

  def save(self, fp:Path=None):
    fp = fp or self.fp
    assert fp, "no path to save"
    faiss.write_index(self.index, str(fp))

  def add(self, v:ndarray):
    assert isinstance(v, ndarray), 'v should be ndarray'
    assert len(v.shape) == 2, 'v.shape shoudl be [N, D]'
    assert self.dim == v.shape[-1], f"dim mismatch: v.dim ({v.shape[-1]}) != vdb.dim ({self.dim})"
    self.index.add(v)

  def search(self, v:ndarray, k:int=10) -> Tuple[ndarray, ndarray]:
    assert self.dim == v.shape[-1], f"dim mismatch: v.dim ({v.shape[-1]}) != vdb.dim ({self.dim})"
    dists, indexes = self.index.search(v, k)
    return dists, indexes

  def query(self, indexes:ndarray) -> ndarray:
    if len(indexes.shape) == 2:
      indexes = indexes.squeeze(axis=-1)
    assert len(indexes.shape) == 1
    return self.index.reconstruct_batch(indexes)

  def nearest_neighbour(self, v:ndarray) -> ndarray:
    assert self.dim == v.shape[-1], f"dim mismatch: v.dim ({v.shape[-1]}) != vdb.dim ({self.dim})"
    dists, indexes, vectors = self.index.search_and_reconstruct(v, k=1)
    return vectors.squeeze(axis=1)


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
    vec_db = VectorDB(dim, index_fp)
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
