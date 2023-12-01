#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/30 

from PIL import ImageFilter

from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from utils import *


class NIPS17_pair(Dataset):

  def __init__(self, filter:str='none'):
    self.filter = filter

    fps_adv = list(DATA_ADV_PATH.iterdir())
    fps_raw = [DATA_RAW_PATH / fp_adv.name for fp_adv in fps_adv]
    self.fps = list(zip(fps_raw, fps_adv))

  def __len__(self):
    return len(self.fps)

  def __getitem__(self, idx):
    fp_raw, fp_adv = self.fps[idx]
    im_raw = pil_to_npimg(self.apply_filter(load_img(fp_raw))).transpose(2, 0, 1).astype(np.float32) / 255.0
    im_adv = pil_to_npimg(self.apply_filter(load_img(fp_adv))).transpose(2, 0, 1).astype(np.float32) / 255.0
    return im_raw, im_adv

  def apply_filter(self, img:PILImage) -> PILImage:
    if self.filter == 'none': return img
    img_low = img.filter(ImageFilter.GaussianBlur(3))
    if self.filter == 'low': return img_low
    if self.filter == 'high': return npimg_to_pil(minmax_norm(npimg_diff(pil_to_npimg(img), pil_to_npimg(img_low))))


def imshow(X, AX, title=''):
  DX = X - AX
  DX = (DX - DX.min()) / (DX.max() - DX.min())

  grid_X  = make_grid( X).permute([1, 2, 0]).detach().cpu().numpy()
  grid_AX = make_grid(AX).permute([1, 2, 0]).detach().cpu().numpy()
  grid_DX = make_grid(DX).permute([1, 2, 0]).detach().cpu().numpy()
  plt.subplot(131) ; plt.title('X')  ; plt.axis('off') ; plt.imshow(grid_X)
  plt.subplot(132) ; plt.title('AX') ; plt.axis('off') ; plt.imshow(grid_AX)
  plt.subplot(133) ; plt.title('DX') ; plt.axis('off') ; plt.imshow(grid_DX)
  plt.tight_layout()
  plt.suptitle(title)

  mng = plt.get_current_fig_manager()
  mng.window.showMaximized()    # 'QT4Agg' backend
  plt.show()
