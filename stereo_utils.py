#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import re
import sys

def load_pfm(fname, downsample):
  if downsample:
        if not os.path.isfile(fname + '.H.pfm'):
            x, scale = load_pfm(fname, False)
            x = x / 2
            x_ = np.zeros((x.shape[0] // 2, x.shape[1] // 2), dtype=np.float32)
            for i in range(0, x.shape[0], 2):
                for j in range(0, x.shape[1], 2):
                    tmp = x[i:i+2,j:j+2].ravel()
                    x_[i // 2,j // 2] = np.sort(tmp)[1]
            save_pfm(fname + '.H.pfm', x_, scale)
            return x_, scale
        else:
            fname += '.H.pfm'
  color = None
  width = None
  height = None
  scale = None
  endian = None
  
  file = open(fname, 'rb')
  header = file.readline().decode('utf-8').rstrip()
  if header == 'PF':
    color = True    
  elif header == 'Pf':
    color = False
  else:
    raise Exception('Not a PFM file.')
 
  dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
  if dim_match:
    width, height = map(int, dim_match.groups())
  else:
    raise Exception('Malformed PFM header.')
 
  scale = float(file.readline().decode('utf-8').rstrip())
  if scale < 0: # little-endian
    endian = '<'
    scale = -scale
  else:
    endian = '>' # big-endian
 
  data = np.fromfile(file, endian + 'f')
  shape = (height, width, 3) if color else (height, width)
  return np.flipud(np.reshape(data, shape)), scale

def save_pfm(fname, image, scale=1):
  file = open(fname, 'w') 
  color = None
 
  if image.dtype.name != 'float32':
    raise Exception('Image dtype must be float32.')
 
  if len(image.shape) == 3 and image.shape[2] == 3: # color image
    color = True
  elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
    color = False
  else:
    raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')
 
  file.write('PF\n' if color else 'Pf\n')
  file.write('%d %d\n' % (image.shape[1], image.shape[0]))
 
  endian = image.dtype.byteorder
 
  if endian == '<' or endian == '=' and sys.byteorder == 'little':
    scale = -scale
 
  file.write('%f\n' % scale)
 
  np.flipud(image).tofile(file)


