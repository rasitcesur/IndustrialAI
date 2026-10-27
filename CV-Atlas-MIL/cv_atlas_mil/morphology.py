#!/usr/bin/env python3
"""Unsupervised local high-resolution morphology descriptors for casting inspection.

16x16 grid primitives aggregated into 32x32/64x64 neighborhoods.
No image-level label or ROI hint is involved in extraction.
"""
from pathlib import Path
import argparse,zipfile,time
import numpy as np
import cv2

CELL = 16
FEATURES = [
 'bh11_max','bh11_mean','bh21_max','bh21_mean',
 'th11_max','th11_mean','th21_max','th21_mean',
 'grad_max','grad_mean', 'dark_line_h_mean','dark_line_v_mean',
 'dark_local_spread','bright_local_spread',
 'bh11_peak_16','bh11_mean_16','bh11_patch_count', 'bh11_patch_peak_gap',
 'th11_peak_16','th11_mean_16','th11_patch_count',
 'grad_peak_16','grad_mean_16', 'grad_patch_count',
 'bh11_p32_max', 'bh11_p32_mean', 'th11_p32_max','grad_p32_max',
 ]

def grid_16(img, mode='mean'):
  cells=img.reshape(32,16,32,16)
  return cells.mean(axis=(1,3)) if mode=='mean' else cells.max(axis=(1,3))

def reduced_64(x,fn):
  return fn(x.reshape(8,4,8,4),axis=(1,3))

def desc(img):
  if img.shape!=(512,512):raise ValueError(f'512x512 required: {img.shape}')
  im=cv2.GaussianBlur(img,(3,3),.7)
  morph=lambda op,n: cv2.morphologyEx(im,op,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(n,n))).astype(np.float32)
  bh11=morph(cv2.MORPH_BLACKHAT,11);bh21=morph(cv2.MORPH_BLACKHAT,21)
  th11=morph(cv2.MORPH_TOPHAT,11);th21=morph(cv2.MORPH_TOPHAT,21)
  gx=cv2.Sobel(im,cv2.CV_32F,1,0,ksize=3);gy=cv2.Sobel(im,cv2.CV_32F,0,1,ksize=3)
  gm=cv2.magnitude(gx,gy)
  # Oriented line-like darkness, morphology at 11px scale.
  lm_h=cv2.morphologyEx(im,cv2.MORPH_BLACKHAT,cv2.getStructuringElement(cv2.MORPH_RECT,(11,1))).astype(np.float32)
  lm_v=cv2.morphologyEx(im,cv2.MORPH_BLACKHAT,cv2.getStructuringElement(cv2.MORPH_RECT,(1,11))).astype(np.float32)
  xs=[bh11,bh21,th11,th21,gm,lm_h,lm_v]
  gmean=[grid_16(x,'mean') for x in xs]
  gmax=[grid_16(x,'max') for x in xs[:5]]
  # no absolute gray. Scale intensities to a stable gray-level dynamic range.
  a=[]
  for j in [0,1,2,3]:
    a.extend([reduced_64(gmax[j],np.max), reduced_64(gmean[j],np.mean)])
  a.extend([reduced_64(gmax[4],np.max),reduced_64(gmean[4],np.mean),
            reduced_64(gmean[5],np.mean),reduced_64(gmean[6],np.mean),
            reduced_64(gmax[0],np.max)-reduced_64(gmean[0],np.min),
            reduced_64(gmax[2],np.max)-reduced_64(gmean[2],np.min)])
  # The 4x4 sub-grid of 16x16 cells within each 64x64 region.
  def local_stats(mx,me,threshold):
    blockmax=mx.reshape(8,4,8,4).transpose(0,2,1,3).reshape(64,16)
    blockmean=me.reshape(8,4,8,4).transpose(0,2,1,3).reshape(64,16)
    peak=blockmax.max(axis=1)
    a.extend([peak.reshape(8,8),blockmean.mean(axis=1).reshape(8,8),
      (blockmax>threshold).sum(axis=1).reshape(8,8),
    ])
    return blockmax
  n=local_stats(gmax[0],gmean[0],20.0)
  a.append((n.max(axis=1)-np.median(n,axis=1)).reshape(8,8))
  local_stats(gmax[2],gmean[2],20.0)
  local_stats(gmax[4],gmean[4],60.0)
  # 32x32 neighborhood maxima aggregate each 2x2 cells to avoid 64x64 averaging.
  for x in [gmax[0],gmean[0],gmax[2],gmax[4]]:
    m=x.reshape(16,2,16,2).max(axis=(1,3))
    a.append(m.reshape(8,2,8,2).max(axis=(1,3)))
  d=np.stack(a,axis=-1).reshape(64,-1).astype(np.float32)
  assert d.shape==(64,len(FEATURES)),(d.shape,len(FEATURES))
  return np.nan_to_num(d/64.0,nan=0,posinf=0,neginf=0)

def extract(zip_path, names,output):
  t=time.perf_counter();n=len(names)
  output=Path(output)
  if output.exists():
    a=np.load(output);cached=a['features'];cn=a['names'];
    if cached.shape==(n,64,len(FEATURES)) and np.array_equal(cn,names):return cached
  features=np.empty((n,64,len(FEATURES)),dtype=np.float32)
  with zipfile.ZipFile(zip_path) as z:
    lookup={v.rsplit('/',1)[-1]:v for v in z.namelist() if v.endswith('.jpeg')}
    for i,name in enumerate(names):
      fname=name.rsplit('/',1)[-1]
      im=cv2.imdecode(np.frombuffer(z.read(lookup[fname]),dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
      features[i]=desc(im)
      if (i+1)%200==0:print('EXTRACTED',i+1,'/',n,'s',round(time.perf_counter()-t,1),flush=True)
  np.savez_compressed(output,features=features,names=names,feature_names=FEATURES)
  return features

if __name__=='__main__':
  p=argparse.ArgumentParser();p.add_argument('--dataset-zip',required=True);p.add_argument('--base-npz',required=True);p.add_argument('--output',required=True);args=p.parse_args()
  a=np.load(args.base_npz);extract(args.dataset_zip,a['names'],args.output)
