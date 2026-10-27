#!/usr/bin/env python3
"""Train all five CV Atlas MIL experts from labeled images. Fitting is not evaluation."""
import argparse,json
from pathlib import Path
import joblib
from cv_atlas_mil import CVAtlasMIL, FUSION_WEIGHTS, __version__
from cv_atlas_mil.data import load_dataset

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True,help='Folder containing def_front and ok_front')
    p.add_argument('--model',type=Path,default=Path('models/cv_atlas_mil.joblib'))
    p.add_argument('--cache',type=Path,default=Path('cache/features.npz'))
    p.add_argument('--refresh-cache',action='store_true')
    p.add_argument('--batch',type=int,default=16)
    p.add_argument('--backend',choices=['auto','cuda','cython','python'],default='auto')
    p.add_argument('--trees',type=int,default=400)
    p.add_argument('--mil-epochs',type=int,default=35)
    p.add_argument('--workers',type=int,default=3)
    p.add_argument('--seed',type=int,default=20261025)
    args=p.parse_args()
    X,G,H,y,names=load_dataset(args.data_dir,args.cache,args.backend,batch=args.batch,refresh_cache=args.refresh_cache)
    model=CVAtlasMIL(trees=args.trees,mil_epochs=args.mil_epochs,
                   workers=args.workers,seed=args.seed).fit(X,G,H,y)
    artifact={'version':__version__,'model':model,'weights':dict(FUSION_WEIGHTS),
             'threshold':.5,'classes':{'defective':0,'ok':1},
             'image_size':[512,512],'training_images':len(y),
             'note':'Training-set fit; independent generalization is not measured by this operation.'}
    args.model.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(artifact,args.model,compress=3)
    print(json.dumps({'saved':str(args.model),'training_images':len(y),
      'healthy_atlas_references':model.original.atlas.reference_count(),
      'weights':FUSION_WEIGHTS},indent=2))

if __name__=='__main__':main()
