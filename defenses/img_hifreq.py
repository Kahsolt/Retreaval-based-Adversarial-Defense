#!/usr/bin/env python3
# Author: Shoi
# Create Time: 2024/02/03


from utils import *
from pywt import dwt2


def img_hifreq_by_unsharp_mask(img:PILImage, R:int) -> npimg_hi:
  
  img_usm = img.filter(ImageFilter.UnsharpMask(R,100,3))
  
  img_usm = pil_to_npimg(img_usm)
  im_raw  = pil_to_npimg(img)

  im_raw_hi = minmax_norm(npimg_diff(img_usm, im_raw))
  # im_raw_lo = minmax_norm(npimg_diff(im_raw, im_raw_hi))

  # plt.imshow(im_raw_lo)     ; plt.axis('off')
  # plt.show()

  return im_raw_hi


def img_hifreq_by_fft(img:PILImage, R:int) -> npimg_hi:
  im_raw = pil_to_npimg(img)
  h,w = im_raw.shape[:2]
  
  lpf = np.zeros((h,w,3))
  for x in range(w):
    for y in range(h):
        if ((x-(w-1)/2)**2 + (y-(h-1)/2)**2) < (R**2):
            lpf[y,x,:] = 1
  hpf = 1-lpf

  freq = np.fft.fft2(im_raw,axes=(0,1))
  freq = np.fft.fftshift(freq)
  lf = freq * lpf
  hf = freq * hpf

  # freq_view = np.log(1 +np.abs(freq))
  # freq_view = (freq_view - freq_view.min()) / (freq_view.max() - freq_view.min()) * 255
  # freq_view = freq_view.astype('uint8').copy()


  img_l = np.abs(np.fft.ifft2(lf,axes=(0,1)))
  img_l = np.clip(img_l,0,255) 
  img_l = minmax_norm(img_l)

  img_h = np.abs(np.fft.ifft2(hf,axes=(0,1)))
  img_h = np.clip(img_h,0,255) 
  img_h = img_h.astype('uint8')

  # plt.imshow(freq_view)     ; plt.axis('off')
  # plt.show()

  return img_h


def img_hifreq_by_dwt(img:PILImage, args:Namespace) -> npimg_hi:
  r, g, b = img.split()
  
  cA1 = dwt2(r, 'haar')[0]
  cA2 = dwt2(g, 'haar')[0]
  cA3 = dwt2(b, 'haar')[0]

  raw_size = (224, 224)
  r_lo = Image.fromarray(cA1).convert('L').resize(raw_size)
  g_lo = Image.fromarray(cA2).convert('L').resize(raw_size)
  b_lo = Image.fromarray(cA3).convert('L').resize(raw_size)

  new = Image.merge('RGB', (r_lo, g_lo, b_lo))

  img_lo = pil_to_npimg(new)
  img_raw = pil_to_npimg(img)
  img_hi = minmax_norm(npimg_diff(img_raw, img_lo))

  # plt.imshow(img_lo)     ; plt.axis('off')
  # plt.show()


  return img_hi


def img_hifreq_by(method:str, img:PILImage, args:Namespace):
  return globals()[f'img_hifreq_by_{method}'](img, args)



if __name__ == '__main__':
  from plot import *
  TYPE = ["unsharp_mask","fft","dwt"]
  FILE = ".\dataset\ILSVRC2012_val_00000031.png"
  image = Image.open(FILE)
  img_h = img_hifreq_by(TYPE[2],image,10)

  plt.subplot(221) ; plt.imshow(image)     ; plt.axis('off') ; plt.title('X')
  plt.subplot(222) ; plt.imshow(img_h)     ; plt.axis('off') ; plt.title('AX')
  plt.tight_layout()
  plt.show()
  # unitest