#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/10/26

# https://blog.csdn.net/qq_42951560/article/details/115456471

from PIL import ImageFilter
from PIL.Image import Image as PILImage

import matplotlib.pyplot as plt
from niqe.niqe import get_niqe
from skimage.metrics import *

from utils import *


def plot_cmp(img_raw:PILImage, img_adv:PILImage):
  img_raw_lo = img_raw.filter(ImageFilter.GaussianBlur(3))
  img_adv_lo = img_adv.filter(ImageFilter.GaussianBlur(3))
  im_raw_lo = pil_to_npimg(img_raw_lo)
  im_adv_lo = pil_to_npimg(img_adv_lo)
  im_raw = pil_to_npimg(img_raw)
  im_adv = pil_to_npimg(img_adv)
  im_raw_hi = im_raw - im_raw_lo
  im_adv_hi = im_adv - im_adv_lo

  print('[metrics]')
  niqe_raw = get_niqe(im_raw) ; print('  niqe_raw:', niqe_raw)
  niqe_adv = get_niqe(im_adv) ; print('  niqe_adv:', niqe_adv)
  mse  = mean_squared_error     (im_raw, im_adv) ; print('  mse:', mse)
  rmse = normalized_root_mse    (im_raw, im_adv) ; print('  rmse:', rmse)
  psnr = peak_signal_noise_ratio(im_raw, im_adv) ; print('  psnr:', psnr)
  ssim = structural_similarity  (to_gray(im_raw), to_gray(im_adv)) ; print('  ssim:', ssim)

  dx    = npimg_abs_diff(im_raw,    im_adv,    name='dx')
  dx_lo = npimg_abs_diff(im_raw_lo, im_adv_lo, name='dx_lo')
  dx_hi = npimg_abs_diff(im_raw_hi, im_adv_hi, name='dx_hi')
  dx0    = minmax_norm(dx, vmin=0, vmax=16)   # eps=16/255
  dx0_lo = minmax_norm(dx_lo)
  dx0_hi = minmax_norm(dx_hi)
  dx1    = minmax_norm(to_ch_avg(dx), vmin=0, vmax=16)
  dx1_lo = minmax_norm(to_ch_avg(dx_lo))
  dx1_hi = minmax_norm(to_ch_avg(dx_hi))

  plt.clf()
  plt.subplot(3, 4,  1) ; plt.axis('off') ; plt.title('X')      ; plt.imshow(im_raw)
  plt.subplot(3, 4,  2) ; plt.axis('off') ; plt.title('AX')     ; plt.imshow(im_adv)
  plt.subplot(3, 4,  3) ; plt.axis('off') ; plt.title('DX0')    ; plt.imshow(dx0)
  plt.subplot(3, 4,  4) ; plt.axis('off') ; plt.title('DX1')    ; plt.imshow(dx1)
  plt.subplot(3, 4,  5) ; plt.axis('off') ; plt.title('X_lo')   ; plt.imshow(im_raw_lo)
  plt.subplot(3, 4,  6) ; plt.axis('off') ; plt.title('AX_lo')  ; plt.imshow(im_adv_lo)
  plt.subplot(3, 4,  7) ; plt.axis('off') ; plt.title('DX0_lo') ; plt.imshow(dx0_lo)
  plt.subplot(3, 4,  8) ; plt.axis('off') ; plt.title('DX1_lo') ; plt.imshow(dx1_lo)
  plt.subplot(3, 4,  9) ; plt.axis('off') ; plt.title('X_hi')   ; plt.imshow(im_raw_hi)
  plt.subplot(3, 4, 10) ; plt.axis('off') ; plt.title('AX_hi')  ; plt.imshow(im_adv_hi)
  plt.subplot(3, 4, 11) ; plt.axis('off') ; plt.title('DX0_hi') ; plt.imshow(dx0_hi)
  plt.subplot(3, 4, 12) ; plt.axis('off') ; plt.title('DX1_hi') ; plt.imshow(dx1_hi)
  plt.show()


def run():
  for fp_adv in DATA_ADV_PATH.iterdir():
    fp_raw = DATA_RAW_PATH / fp_adv.name

    img_raw = load_img(fp_raw)
    img_adv = load_img(fp_adv)
    plot_cmp(img_raw, img_adv)


if __name__ == '__main__':
  run()
