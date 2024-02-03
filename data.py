#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/30 

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

from utils import *


class ImageNet_1k(Dataset):

  def __init__(self, root:Path=DATA_IMAGENET_1K_PATH, limit:int=-1, shuffle:bool=False):
    with open(root / 'image_name_to_class_id_and_name.json', encoding='utf-8') as fh:
      mapping = json.load(fh)
    fps = list((root / 'val').iterdir())
    tgts = [mapping[fp.name]['class_id'] for fp in fps]

    self.metadata = [x for x in zip(fps, tgts)]
    if shuffle: random.shuffle(self.metadata)
    if limit > 0: self.metadata = self.metadata[:limit]

  def __len__(self):
    return len(self.metadata)

  def __getitem__(self, idx):
    fp, tgt = self.metadata[idx]
    im = hwc2chw(load_im(fp, 'f32'))
    return im, tgt


class NIPS17_pair(Dataset):

  def __init__(self, filter:str='none'):
    self.filter = filter

    fps_adv = list(DATA_NIPS17_ADV_PATH.iterdir())
    fps_raw = [DATA_NIPS17_RAW_PATH / fp_adv.name for fp_adv in fps_adv]
    self.fps = list(zip(fps_raw, fps_adv))

  def __len__(self):
    return len(self.fps)

  def __getitem__(self, idx):
    fp_raw, fp_adv = self.fps[idx]
    im_raw = hwc2chw(pil_to_npimg(self.apply_filter(load_img(fp_raw)), 'f32'))
    im_adv = hwc2chw(pil_to_npimg(self.apply_filter(load_img(fp_adv)), 'f32'))
    return im_raw, im_adv

  def apply_filter(self, img:PILImage) -> PILImage:
    if self.filter == 'none': return img
    img_low = img.filter(ImageFilter.GaussianBlur(3))
    if self.filter == 'low': return img_low
    if self.filter == 'high': return npimg_to_pil(minmax_norm(npimg_diff(pil_to_npimg(img), pil_to_npimg(img_low))))


def normalize(X: Tensor) -> Tensor:
  ''' NOTE: to insure attack validity, normalization is delayed until put into model '''

  mean = (0.485, 0.456, 0.406)
  std  = (0.229, 0.224, 0.225)
  return TF.normalize(X, mean, std)       # [B, C, H, W]
