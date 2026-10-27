"""Deployable high-resolution morphology expert (trained only from image labels).

Values: defective=0, OK=1. Accepts precomputed 64x28 morphology descriptors
from cv_atlas_mil.morphology.desc or from a cached extractor.
"""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

class HighResolutionMorphologyExpert:
    def __init__(self, trees=400, seed=2026, workers=4):
        self.trees = trees
        self.seed = seed
        self.workers = workers

    def fit(self, features: np.ndarray, labels: np.ndarray):
        f=np.asarray(features,dtype=np.float32)
        y=np.asarray(labels,dtype=int)
        if f.ndim != 3 or f.shape[1:] != (64,28):
            raise ValueError('Features must have shape (N,64,28)')
        if len(f)!=len(y) or set(np.unique(y))!={0,1}:
            raise ValueError('Image labels must contain 0 (defective), 1 (OK)')
        healthy=f[y==1]
        self.med_=np.median(healthy,axis=0).astype(np.float32)
        q10=np.quantile(healthy,.10,axis=0)
        q90=np.quantile(healthy,.90,axis=0)
        self.scale_=np.maximum(q90-q10,.025).astype(np.float32)
        x=self._transform(f)
        self.model_=ExtraTreesClassifier(n_estimators=self.trees,max_features=128,
                 min_samples_leaf=1,bootstrap=False,random_state=self.seed,n_jobs=self.workers)
        self.model_.fit(x,y)
        self.train_n_=len(y)
        return self

    def _transform(self,features):
        f=np.asarray(features,np.float32)
        if f.ndim != 3 or f.shape[1:] != (64,28):raise ValueError('Expect (N,64,28) features')
        z=np.clip((f-self.med_)/self.scale_,-8,8)
        high=np.maximum(z,0.)
        summary=np.concatenate([high.max(axis=1),np.quantile(high,.95,axis=1),high.mean(axis=1)],axis=1)
        return np.ascontiguousarray(np.concatenate([f.reshape(len(f),-1),z.reshape(len(f),-1),summary],axis=1),dtype=np.float32)

    def predict_defect_probability(self, features):
        p=self.model_.predict_proba(self._transform(features))
        return p[:,list(self.model_.classes_).index(0)]
