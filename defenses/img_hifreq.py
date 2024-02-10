#!/usr/bin/env python3
# Author: Shoi
# Create Time: 2024/02/03

from numpy.fft import fft2, fftshift, ifft2, ifftshift
from pywt import dwt2

from utils import *

DEBUG_PLOT = False


def img_hifreq_by_unsharp_mask(img:PILImage, args:Namespace) -> npimg_hi:
  R: int = args.radius
  
  img_lo = img.filter(ImageFilter.GaussianBlur(R))
  im_lo = pil_to_npimg(img_lo)

  im_raw = pil_to_npimg(img)
  im_hi = minmax_norm(npimg_diff(im_raw, im_lo))

  if DEBUG_PLOT:
    plt.imshow(im_hi) ; plt.axis('off')
    plt.show()

  return im_hi, im_lo


def img_hifreq_by_fft(img:PILImage, args:Namespace) -> npimg_hi:
  R: int = args.radius

  im_raw = pil_to_npimg(img)
  h, w = im_raw.shape[:2]
  
  lpf = np.zeros_like(im_raw, dtype=np.int8)
  for x in range(w):
    for y in range(h):
        if ((x-(w-1)/2)**2 + (y-(h-1)/2)**2) < (R**2):
            lpf[y,x,:] = 1
  hpf = 1 - lpf

  freq = fft2(im_raw, axes=(0,1))
  freq = fftshift(freq)

  if DEBUG_PLOT:
    freq_view = np.log1p(np.abs(freq))
    freq_view = minmax_norm(freq_view)
    plt.imshow(freq_view) ; plt.axis('off')
    plt.show()

  lf = freq * lpf
  img_l = np.abs(ifft2(ifftshift(lf), axes=(0, 1)))
  img_l = np.clip(img_l, 0, 255) 
  img_l = minmax_norm(img_l)

  hf = freq * hpf
  img_h = np.abs(ifft2(ifftshift(hf), axes=(0, 1)))
  img_h = np.clip(img_h, 0, 255) 
  img_h = img_h.astype(np.uint8)

  return img_h, img_l


def img_hifreq_by_dwt(img:PILImage, args:Namespace) -> npimg_hi:
  r, g, b = img.split()
  cA1 = dwt2(r, 'haar')[0]
  cA2 = dwt2(g, 'haar')[0]
  cA3 = dwt2(b, 'haar')[0]

  r_lo = Image.fromarray(cA1).convert('L').resize(img.size)
  g_lo = Image.fromarray(cA2).convert('L').resize(img.size)
  b_lo = Image.fromarray(cA3).convert('L').resize(img.size)
  new = Image.merge('RGB', (r_lo, g_lo, b_lo))

  img_lo = pil_to_npimg(new)
  img_raw = pil_to_npimg(img)
  img_hi = minmax_norm(npimg_diff(img_raw, img_lo))

  if DEBUG_PLOT:
    plt.imshow(img_hi) ; plt.axis('off')
    plt.show()

  return img_hi, img_lo


def img_hifreq_by(method:str, img:PILImage, args:Namespace):
  return globals()[f'img_hifreq_by_{method}'](img, args)


if __name__ == '__main__':
  from argparse import ArgumentParser
  from plot import *

  parser = ArgumentParser()
  parser.add_argument('-M', '--method', default='unsharp_mask', choices=['unsharp_mask', 'fft', 'dwt'], help='highfreq extraction algorithm')
  parser.add_argument('-R', '--radius', type=int, default=3, help='mask radius in unsharp_mask and fft')
  args = parser.parse_args()

  fp = DATA_IMAGENET_1K_PATH / 'val' / 'ILSVRC2012_val_00000031.png'
  image = load_img(fp)
  img_h, img_l = img_hifreq_by(args.method, image, args)

  plt.subplot(131) ; plt.imshow(image) ; plt.axis('off') ; plt.title('img')
  plt.subplot(132) ; plt.imshow(img_h) ; plt.axis('off') ; plt.title('high')
  plt.subplot(133) ; plt.imshow(img_l) ; plt.axis('off') ; plt.title('low')
  plt.tight_layout()
  plt.show()
