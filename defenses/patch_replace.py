#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/03

from __future__ import annotations

from torch.autograd.function import Function, FunctionCtx

from .base import *
from .vector_db import VectorDB
from .img_hifreq import img_hifreq_by_unsharp_mask

BBox = List[slice]

# fix channel size
opt_C = 3


def make_bboxes(im:ndarray, patch_size:int=16, shift_size:int=None) -> List[BBox]:
  H, W, C = im.shape
  P = patch_size
  S = shift_size or (P // 2)
  SLICE_ALL = slice(None)
  bboxes: List[BBox] = []
  i = 0
  while i + P <= H:
    j = 0
    while j + P <= W:
      bboxes.append((slice(i, i + P), slice(j, j + P), SLICE_ALL))
      j += S
    i += S
  return bboxes


def make_vdb(img_dp:Path, patch_size:int=16, img_to_im_fn:Callable=None) -> VectorDB:
  vdb = VectorDB(patch_size**2 * opt_C)
  for fp in tqdm(list(img_dp.iterdir())):
    img = load_img(fp)
    im = img_to_im_fn(img)
    bboxes = make_bboxes(im, patch_size)
    patches = np.stack([im[bbox] for bbox in bboxes], axis=0)
    patch_vectors = patches.reshape(len(patches), -1)   # [N, P**2*C]
    vdb.add(patch_vectors)
  return vdb


class PatchReplacePassGrad(Function):

  ''' just pass by gradients through this PatchReplace operation :( '''

  @staticmethod
  def forward(ctx:FunctionCtx, X:Tensor, self:PatchReplaceDefense) -> Tensor:
    return self.transform(X)

  @staticmethod
  def backward(ctx:FunctionCtx, grad:Tensor) -> Tensor:
    return grad, None


class PatchReplaceApproxGrad(Function):

  ''' find way (?) to estimate the gradients :) '''

  @staticmethod
  def forward(ctx:FunctionCtx, X:Tensor, self:PatchReplaceDefense) -> Tensor:
    TX = self.transform(X)
    ctx.save_for_backward(X, TX)
    return TX

  @staticmethod
  def backward(ctx:FunctionCtx, grad:Tensor) -> Tensor:
    X, TX = ctx.saved_tensors
    DX = TX - X
    return grad * DX, None


class PatchReplaceDefense(BaseDefense):

  def __init__(self, ref_db:str='NIPS17', patch_size:int=16, order:int=0):
    self.order = order
    self.patch_size = patch_size
    self.shift_size = patch_size // 2

    if order == 0:
      extract_fn = lambda img: pil_to_npimg(img, 'f32')
      apply_fn = lambda img: pil_to_npimg(img, 'f32')
    elif order == 1:
      # hifreq, FIXME: make configurable
      args = Namespace()
      args.radius = 3
      extract_fn = lambda img: img_hifreq_by_unsharp_mask(img, args)[0]
      apply_fn = lambda img: img_hifreq_by_unsharp_mask(img, args)
    else: raise ValueError(f'unknwon order: {order}')
    self.apply_fn = apply_fn

    if ref_db == 'NIPS17':
      index_fp = LOG_PATH / f'NIPS17_ps={patch_size}_f={order}.index'
      img_dp = DATA_NIPS17_RAW_PATH
    elif ref_db == 'ImageNet':
      index_fp = LOG_PATH / f'ImageNet_ps={patch_size}_f={order}.index'
      img_dp = DATA_IMAGENET_1K_PATH / 'val'
    if not index_fp.exists():
      import os
      vdb = make_vdb(img_dp, patch_size, extract_fn)
      vdb.save(index_fp)
      print('>> index filesize:', os.path.getsize(index_fp) / 2**20, 'MB')
    self.vdb = VectorDB.load(index_fp)

  def __call__(self, X:Tensor) -> Tensor:
    return PatchReplacePassGrad.apply(X, self)

  def transform(self, X:Tensor) -> Tensor:
    transform_f = getattr(self, f'transform_f{self.order}')
    return torch.stack([transform_f(x) for x in X], axis=0)

  def transform_f0(self, x:Tensor) -> Tensor:
    im = std_clip(tensor_to_npimg(x))
    #im = self.apply_fn(npimg_to_pil(im))   # ignored, this is identity :)
    im_dfn = self.replace(im)
    x_dfn = npimg_to_tensor(std_clip(im_dfn))

    if not 'show':
      with torch.no_grad():
        diff: Tensor = torch.abs(x_dfn - x)
        print('Linf:', diff.max().item())
        print('L1:', diff.mean().item())
        plot3(x, x_dfn, title='transform_f0')

    return x_dfn

  def transform_f1(self, x:Tensor) -> Tensor:
    im = std_clip(tensor_to_npimg(x))
    im_hi, im_lo = self.apply_fn(npimg_to_pil(im))
    im_hi_rec = self.replace(im_hi)
    im_dfn = im_hi_rec + im_lo
    x_dfn = npimg_to_tensor(std_clip(im_dfn))

    if not 'show hifreq':
      with torch.no_grad():
        diff: ndarray = np.abs(im_hi_rec - im_hi)
        print('Linf:', diff.max())
        print('L1:', diff.mean())
        X_hi = npimg_to_tensor(minmax_norm(im_hi))
        X_hi_pr = npimg_to_tensor(minmax_norm(im_hi_rec))
        plot3(X_hi, X_hi_pr, title='transform_f1')

    return x_dfn

  @timer
  def replace(self, im:ndarray) -> ndarray:
    bboxes = make_bboxes(im, self.patch_size, self.shift_size)
    patches: ndarray = np.stack([im[bbox] for bbox in bboxes], axis=0)
    patch_vectors = patches.reshape(len(patches), -1) # [N, P**2*C]
    replacement_vectors = self.vdb.nearest_neighbour(patch_vectors)
    replacements = replacement_vectors.reshape(len(patches), *patches[0].shape) # [N, P, P, C]

    out = im.copy()
    for i, slicers in enumerate(bboxes):
      out[slicers] = replacements[i]
    return out


if __name__ == '__main__':
  from model import get_model
  from plot import *

  ref_db = 'NIPS17'
  #ref_db = 'ImageNet'

  #fp = DATA_IMAGENET_1K_PATH / 'val' / 'ILSVRC2012_val_00000031.png'
  fp = DATA_NIPS17_ADV_PATH / '0.png'
  im = load_im(fp, 'f32')
  X = npimg_to_tensor(im).unsqueeze(0)

  model = get_model('resnet18')
  with torch.inference_mode():
    logits_X = model(X)
    pred_X = logits_X.argmax(dim=-1)
    prob_X = F.softmax(logits_X, dim=-1)[0, pred_X.item()].item()

  dfn = PatchReplaceDefense(ref_db, patch_size=16, order=1)
  breakpoint()
  s = time()
  BX = dfn(X)
  t = time()
  print(f'>> PatchReplaceDefense time cost: {t - s:.3f}s')

  with torch.inference_mode():
    logits_BX = model(BX)
    pred_BX = logits_BX.argmax(dim=-1)
    prob_BX = F.softmax(logits_BX, dim=-1)[0, pred_BX.item()].item()

  print('[pred]')
  print(f'   pred_X:  {pred_X.item()} ({prob_X:.3%})')
  print(f'   pred_BX: {pred_BX.item()} ({prob_BX:.3%})')

  Linf, L1, L2 = Linf_L1_L2(X, BX)
  print('[dist]')
  print('   Linf:', Linf)
  print('   L1:', L1)
  print('   L2:', L2)

  plot3(X, BX, title='PatchReplaceDefense')
