#!/usr/bin/env python3
"""Measure isolated original 2048-feature extraction latency and CPU parity.

Does not benchmark MIL, atlas, geometric descriptors, disk IO or training.
Usage: python benchmark_backends.py --backend auto --images 32
"""
import argparse
import json
from time import perf_counter
import numpy as np
from cv_atlas_mil.features import base_features_batch, available_backends, selected_backend


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',default='auto',choices=['auto','cuda','cython','python'])
    p.add_argument('--images',type=int,default=16)
    p.add_argument('--repeat',type=int,default=3)
    a=p.parse_args()
    if a.images<1 or a.repeat<1:p.error('positive --images and --repeat required')
    rng=np.random.default_rng(951)
    images=rng.integers(0,256,(a.images,512,512),dtype=np.uint8)
    engine=selected_backend(a.backend)
    # Exclude CUDA first-launch compilation and self-check from steady-state timing.
    base_features_batch(images[:1],backend=engine)
    timings=[]
    for _ in range(a.repeat):
        start=perf_counter()
        out=base_features_batch(images,backend=engine)
        timings.append(perf_counter()-start)
    ref=base_features_batch(images[:min(3,a.images)],backend='python')
    deviation=float(np.max(np.abs(out[:len(ref)]-ref)))
    print(json.dumps({'selected_backend':engine,'capabilities':available_backends(),
          'images':a.images,'latency_seconds_median':float(np.median(timings)),
          'milliseconds_per_image':1000*float(np.median(timings))/a.images,
          'max_absolute_difference_vs_python':deviation,
          'note':'Excludes disk IO, geometry descriptors, atlas, MIL and Extra Trees'},indent=2))

if __name__=='__main__':main()
