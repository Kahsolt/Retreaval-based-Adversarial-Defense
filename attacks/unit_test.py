#!/usr/bin/env python3
# Author: zhou_fanqi
# Create Time: 2024/2/10

from plot import plot3
from utils import *
from model import get_model


def unittest(atk_class, fp:Path=None):
  fp = fp or (DATA_NIPS17_RAW_PATH / '0.png')
  im = load_im(fp, 'f32')
  X = npimg_to_tensor(im).unsqueeze(0)
  
  model = get_model('resnet18')
  with torch.inference_mode():
    outputs = model(X)
    Y = outputs.argmax(dim=-1)
    
  atk = atk_class(model)
  AX = atk(X, Y)
  
  with torch.inference_mode():
    outputs_AX = model(AX)
    pred_AX = outputs_AX.argmax(dim=-1)
    
  print('>>>prediction of X and AX')
  print('pred_X:', Y.item())
  print('pred_AX:', pred_AX.item())
  
  Linf, L1, L2 = Linf_L1_L2(X, AX)
  print('>>>Linf, L1, L2')
  print('Linf:', Linf)
  print('L1:', L1)
  print('L2:', L2)
  
  plot3(X, AX, title=atk_class)
  