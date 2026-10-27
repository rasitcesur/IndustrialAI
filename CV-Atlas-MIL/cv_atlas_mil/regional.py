"""Spatially co-located 64x64 descriptors and image-supervised regional MIL.

Local evidence is ordered identically for geometry/intensity, high-resolution
morphology, and the healthy-reference atlas. Labels apply to whole images only.
"""
from __future__ import annotations

import numpy as np
from .mil import train_model, infer_model, from_numpy, state_to_numpy


def fit_normalizer(values: np.ndarray, labels: np.ndarray, lower: float,
                   upper: float, floor: float) -> tuple[np.ndarray,np.ndarray]:
    normals = np.asarray(values, np.float32)[np.asarray(labels)==1]
    if len(normals)==0:
        raise ValueError('No healthy training images')
    center = np.median(normals, axis=0).astype(np.float32)
    spread = np.maximum(np.quantile(normals,upper,axis=0) -
                        np.quantile(normals,lower,axis=0),floor).astype(np.float32)
    return center,spread


def co_located_vectors(geom: np.ndarray, morph: np.ndarray, atlas_regions: np.ndarray,
                       geom_normalizer: tuple[np.ndarray,np.ndarray],
                       morph_normalizer: tuple[np.ndarray,np.ndarray]) -> np.ndarray:
    """Each region r: [G_r (14), H_r (28), A_r (4), row/col (2)] = 48."""
    geom = np.asarray(geom,np.float32)
    morph = np.asarray(morph,np.float32)
    atlas_regions = np.asarray(atlas_regions,np.float32)
    n=len(geom)
    if geom.shape!=(n,64,14) or morph.shape!=(n,64,28) or atlas_regions.shape!=(n,64,4):
        raise ValueError('Expected (N,64,14), (N,64,28), (N,64,4)')
    zg=np.clip((geom-geom_normalizer[0])/geom_normalizer[1],-8,8).astype(np.float32)
    zh=np.clip((morph-morph_normalizer[0])/morph_normalizer[1],-8,8).astype(np.float32)
    za=np.log1p(np.maximum(atlas_regions,0)).astype(np.float32)
    row=np.repeat(np.arange(8),8).astype(np.float32)/3.5 - 1
    col=np.tile(np.arange(8),8).astype(np.float32)/3.5 - 1
    xy=np.broadcast_to(np.stack([row,col],axis=1),(n,64,2))
    return np.ascontiguousarray(np.concatenate([zg,zh,za,xy],axis=-1),np.float32)


class RegionalJointMIL:
    def __init__(self,epochs=35,seed=20261025,torch_threads=2):
        self.epochs=epochs
        self.seed=seed
        self.torch_threads=torch_threads

    def fit(self,geom,morph,atlas_regions,labels):
        self.gnorm=fit_normalizer(geom,labels,.15,.85,.04)
        self.hnorm=fit_normalizer(morph,labels,.10,.90,.025)
        x=co_located_vectors(geom,morph,atlas_regions,self.gnorm,self.hnorm)
        model=train_model(x,labels,epochs=self.epochs,seed=self.seed,threads=self.torch_threads)
        self.state=state_to_numpy(model)
        return self

    def predict_scores(self,geom,morph,atlas_regions,explain=False):
        x=co_located_vectors(geom,morph,atlas_regions,self.gnorm,self.hnorm)
        p,att=infer_model(from_numpy(self.state,d=48),x)
        return (p,att) if explain else p
