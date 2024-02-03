#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/03

import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from utils import *


def plot3(X:Tensor, AX:Tensor, title:str='', fp:Path=None):
  DX = AX - X
  DX = (DX - DX.min()) / (DX.max() - DX.min())

  grid_X  = make_grid( X).permute([1, 2, 0]).detach().cpu().numpy()
  grid_AX = make_grid(AX).permute([1, 2, 0]).detach().cpu().numpy()
  grid_DX = make_grid(DX).permute([1, 2, 0]).detach().cpu().numpy()
  plt.subplot(131) ; plt.title('X')  ; plt.axis('off') ; plt.imshow(grid_X)
  plt.subplot(132) ; plt.title('AX') ; plt.axis('off') ; plt.imshow(grid_AX)
  plt.subplot(133) ; plt.title('DX') ; plt.axis('off') ; plt.imshow(grid_DX)
  plt.tight_layout()
  plt.suptitle(title)

  if fp:
    plt.savefig(fp, dpi=600)
  else:
    try:
      mng = plt.get_current_fig_manager()
      mng.window.showMaximized()    # for 'QT4Agg' backend
    except: pass
    plt.show()
