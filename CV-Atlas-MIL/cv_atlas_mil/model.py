"""CV Atlas MIL: five experts and fixed inference-time score fusion.

Fusion scores are all P(defective), where label 0 is defective and 1 is OK.
No pixel-level annotation, segmentation, geometric registration, or runtime
weight optimization is performed. Scores are research outputs, not calibrated
probabilities guaranteed under acquisition shift.
"""
from __future__ import annotations
import numpy as np
from .pipeline import CoreAtlasMIL
from .regional import RegionalJointMIL
from .morphology_model import HighResolutionMorphologyExpert

FUSION_WEIGHTS = {'base':.45,'atlas':.10,'mil':.05,'regional':.05,'morphology':.35}

class CVAtlasMIL:
    def __init__(self,trees=400,reference_k=24,mil_epochs=35,seed=20261025,workers=3,torch_threads=2):
        self.trees=int(trees)
        self.reference_k=int(reference_k)
        self.mil_epochs=int(mil_epochs)
        self.seed=int(seed)
        self.workers=int(workers)
        self.torch_threads=int(torch_threads)

    def fit(self,X,G,H,y):
        X=np.asarray(X,np.float32)
        G=np.asarray(G,np.float32)
        H=np.asarray(H,np.float32)
        y=np.asarray(y,dtype=int)
        if X.shape!=(len(y),128) or G.shape!=(len(y),64,14) or H.shape!=(len(y),64,28):
            raise ValueError('Expected X=(N,128), G=(N,64,14), H=(N,64,28)')
        if set(np.unique(y))!={0,1}:
            raise ValueError('Both image classes are needed')
        self.original=CoreAtlasMIL(trees=self.trees,reference_k=self.reference_k,
                        mil_epochs=self.mil_epochs,seed=self.seed,
                        workers=self.workers,torch_threads=self.torch_threads).fit(X,G,y)
        _,train_regions=self.original.atlas.transform(G,exclude_training_self=True,return_region=True)
        self.regional=RegionalJointMIL(epochs=self.mil_epochs,seed=self.seed,
                        torch_threads=self.torch_threads).fit(G,H,train_regions,y)
        self.morphology=HighResolutionMorphologyExpert(trees=self.trees,seed=self.seed,
                        workers=self.workers).fit(H,y)
        self.training_size=len(y)
        return self

    def predict_scores(self,X,G,H,explain=False):
        X=np.asarray(X,np.float32)
        G=np.asarray(G,np.float32)
        H=np.asarray(H,np.float32)
        if X.shape!=(len(G),128) or G.shape!=(len(H),64,14) or H.shape!=(len(X),64,28):
            raise ValueError('Bad input shapes')
        prior=self.original.predict_scores(X,G,explain=explain)
        _,regional_atlas=self.original.atlas.transform(G,return_region=True)
        if explain:
            regional,regional_attention=self.regional.predict_scores(G,H,regional_atlas,explain=True)
        else:
            regional=self.regional.predict_scores(G,H,regional_atlas)
        morphology=self.morphology.predict_defect_probability(H)
        scores=dict(base=prior['base'],atlas=prior['atlas'],mil=prior['mil'],
                    regional=regional,morphology=morphology)
        blended=np.clip(sum(FUSION_WEIGHTS[k]*scores[k] for k in FUSION_WEIGHTS),0.0,1.0)
        scores['prob_defect']=blended
        if explain:
            scores['region_attention']=prior['region_attention']
            scores['region_deviation']=prior['region_deviation']
            scores['joint_attention']=regional_attention
        return scores

    def predict(self,X,G,H,threshold=.5):
        if not 0<=threshold<=1: raise ValueError('threshold must be in [0,1]')
        p=self.predict_scores(X,G,H)['prob_defect']
        return np.where(p>=threshold,0,1)
