"""Normal atlas: position-conditioned per-region comparison to healthy images.

No geometric registration is performed. Camera pose invariance is NOT assumed.
Gallery references and normalization statistics are FIT ONLY ON TRAIN normals.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist

FLOORS=np.array([.035,.035,.035,.06,.035,.035,.035,.035,.06,.035,.06,.10,.12,.09],np.float32)


def global_view(G):
    return np.concatenate((G[:,:,13],.6*G[:,:,12]),axis=1).astype(np.float32)


class ReferenceAtlas:
    def __init__(self,k=24):
        self.k=int(k)

    def fit(self,G,y):
        G=np.asarray(G,np.float32);y=np.asarray(y)
        if G.ndim!=3 or G.shape[1:]!=(64,14):
            raise ValueError('Expected geometry array (N,64,14)')
        normals=np.flatnonzero(y==1)
        if len(normals)<4:
            raise ValueError('At least four healthy training images required')
        H=G[normals]
        self.scale=np.maximum(np.quantile(H,.85,axis=0)-np.quantile(H,.15,axis=0),FLOORS[None,:]).astype(np.float32)
        self.F=global_view(G)
        ref=self.F[normals]
        self.gscale=np.maximum(np.quantile(ref,.85,axis=0)-np.quantile(ref,.15,axis=0),.075).astype(np.float32)
        self.R=H.copy()
        self.Fref=(ref/self.gscale).astype(np.float32)
        self.gallery_train_indices=normals.copy()
        self.k_eff=min(self.k,len(normals)-1)
        return self

    def transform(self,G,exclude_training_self=False,chunk=72,return_region=False):
        """G: query descriptors; self exclusion only if query is EXACT training G in same order."""
        G=np.asarray(G,dtype=np.float32)
        if G.ndim!=3 or G.shape[1:]!=(64,14):
            raise ValueError('Expected (N,64,14)')
        if exclude_training_self and len(G)!=len(self.F):
            raise ValueError('Self exclusion requires training samples in original order')
        result=[];locals_=[]
        for st in range(0,len(G),chunk):
            Q=G[st:st+chunk]
            d=cdist(global_view(Q)/self.gscale,self.Fref,'sqeuclidean').astype(np.float32)
            if exclude_training_self:
                ids=np.arange(st,st+len(Q))
                d[ids[:,None]==self.gallery_train_indices[None,:]]=np.inf
            # Match original CV's top-k selection.
            j=np.argpartition(d,self.k_eff-1,axis=1)[:,:self.k_eff]
            refs=self.R[j]
            diff=np.clip((Q[:,None,:,:]-refs)/self.scale[None,None,:,:],-12,12)
            geo=np.sqrt(np.mean(diff[:,:,:,:10]**2,axis=3))
            intensity=np.sqrt(np.mean(diff[:,:,:,10:]**2,axis=3))
            joint=.65*geo+.35*intensity
            best=np.argmin(joint,axis=1)
            row=np.arange(len(Q))[:,None]
            region_id=np.arange(64)[None,:]
            gbest=geo[row,best,region_id]
            ibest=intensity[row,best,region_id]
            nearest=joint[row,best,region_id]
            kth=min(2,self.k_eff-1)
            third=np.partition(joint,kth,axis=1)[:,kth,:]
            local=np.stack((gbest,ibest,nearest,third),axis=2).astype(np.float32)
            a=nearest.reshape(-1,8,8)
            summary=np.array([
                a.mean(axis=(1,2)), a.max(axis=(1,2)),
                np.quantile(a,.95,axis=(1,2)),np.quantile(a,.75,axis=(1,2)),
                np.mean(a>1.5,axis=(1,2)),np.mean(a>2.,axis=(1,2)),
                np.mean(np.minimum(a[:,1:,:],a[:,:-1,:]),axis=(1,2)),
                np.mean(np.minimum(a[:,:,1:],a[:,:,:-1]),axis=(1,2))
            ],np.float32).T
            features=np.concatenate([local[:,:,i] for i in range(4)]+[summary],axis=1).astype(np.float32)
            result.append(features)
            if return_region:locals_.append(local)
        if not result:
            raise ValueError('Empty batch')
        return (np.concatenate(result),np.concatenate(locals_)) if return_region else np.concatenate(result)

    def reference_count(self):
        return len(self.R)


def normal_profile(G,y):
    H=np.asarray(G,dtype=np.float32)[np.asarray(y)==1]
    if len(H)==0:raise ValueError('No healthy samples')
    center=np.median(H,axis=0)
    scale=np.maximum(np.quantile(H,.85,axis=0)-np.quantile(H,.15,axis=0),.04)
    return center.astype(np.float32),scale.astype(np.float32)
