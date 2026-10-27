"""Three experts with FIXED late fusion weights; no test-time gating or registration."""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from .atlas import ReferenceAtlas,normal_profile
from .mil import train_model, infer_model, vectors, from_numpy, state_to_numpy

WEIGHTS={'base':0.80,'atlas':0.15,'mil':0.05}


def tree(features,trees,workers,seed=2026):
    return ExtraTreesClassifier(n_estimators=trees,max_features=features,
        min_samples_leaf=1,random_state=seed,n_jobs=workers,bootstrap=False)


def defective_probability(estimator,x):
    classes=estimator.classes_.tolist()
    if 0 not in classes:raise ValueError('Classifier missing defective class (0)')
    return estimator.predict_proba(x)[:,classes.index(0)]


class CoreAtlasMIL:
    def __init__(self,trees=400,reference_k=24,mil_epochs=35,seed=20261025,workers=3,torch_threads=2):
        self.trees=trees; self.reference_k=reference_k
        self.mil_epochs=mil_epochs;self.seed=seed
        self.workers=workers;self.torch_threads=torch_threads

    def fit(self,X,G,y):
        X=np.asarray(X,np.float32);G=np.asarray(G,np.float32);y=np.asarray(y,dtype=int)
        if X.ndim!=2 or X.shape[1]!=128 or G.shape!=(len(y),64,14):
            raise ValueError('X must have shape (N,128), G must have shape (N,64,14)')
        if set(np.unique(y))!={0,1}:
            raise ValueError('Expected both classes: defective=0 and healthy=1')
        self.base=tree(128,self.trees,self.workers).fit(X,y)
        self.atlas=ReferenceAtlas(k=self.reference_k).fit(G,y)
        atlas_features,regions=self.atlas.transform(G,exclude_training_self=True,return_region=True)
        self.atlas_clf=tree(64,self.trees,self.workers).fit(atlas_features,y)
        self.profile=normal_profile(G,y)
        mil_input=vectors(G,regions,self.profile)
        model=train_model(mil_input,y,self.mil_epochs,self.seed,threads=self.torch_threads)
        self.mil_state=state_to_numpy(model)
        self.train_n=len(y)
        self.train_counts={int(k):int((y==k).sum()) for k in [0,1]}
        return self

    def predict_scores(self,X,G,explain=False):
        X=np.asarray(X,np.float32);G=np.asarray(G,np.float32)
        if X.ndim!=2 or X.shape[1]!=128 or G.shape!=(len(X),64,14):
            raise ValueError('Bad feature shapes')
        pb=defective_probability(self.base,X)
        af,regions=self.atlas.transform(G,return_region=True)
        pa=defective_probability(self.atlas_clf,af)
        model=from_numpy(self.mil_state)
        pm,attention=infer_model(model,vectors(G,regions,self.profile))
        scores=.80*pb+.15*pa+.05*pm
        result={'prob_defect':scores,'base':pb,'atlas':pa,'mil':pm}
        if explain:
            result['region_attention']=attention
            result['region_deviation']=regions[:,:,2]
        return result

    def predict(self,X,G,threshold=.5):
        p=self.predict_scores(X,G)['prob_defect']
        return np.where(p>=threshold,0,1)
