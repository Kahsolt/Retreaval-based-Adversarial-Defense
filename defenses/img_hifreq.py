#!/usr/bin/env python3
# Author: 
# Create Time: 2024/02/03

from utils import *


def img_hifreq_by_unsharp_mask(img:PILImage, args:Namespace) -> npimg_hi:
  pass


def img_hifreq_by_fft(img:PILImage, args:Namespace) -> npimg_hi:
  pass


def img_hifreq_by_dwt(img:PILImage, args:Namespace) -> npimg_hi:
  pass


def img_hifreq_by(method:str, img:PILImage, args:Namespace):
  return globals()[f'img_hifreq_by_{method}'](img, args)



if __name__ == '__main__':
  from plot import *

  # unitest
  pass
