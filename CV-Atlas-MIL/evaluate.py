#!/usr/bin/env python3
"""Full five-expert 5-fold outer-CV fitting on actual images (exploratory)."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from cv_atlas_mil import CVAtlasMIL,FUSION_WEIGHTS
from cv_atlas_mil.data import load_dataset

def _csv(file,rows):
    if not rows:return
    with file.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--cache',type=Path,default=Path('cache/features.npz'))
    p.add_argument('--output',type=Path,default=Path('evaluation'))
    p.add_argument('--refresh-cache',action='store_true')
    p.add_argument('--backend',choices=['auto','cuda','cython','python'],default='auto')
    p.add_argument('--batch',type=int,default=16)
    p.add_argument('--folds',type=int,default=5)
    p.add_argument('--seeds',nargs='+',type=int,default=[20261025,20261026,20261027])
    p.add_argument('--trees',type=int,default=400)
    p.add_argument('--mil-epochs',type=int,default=35)
    p.add_argument('--workers',type=int,default=3)
    p.add_argument('--max-folds',type=int,default=0,help='Debug: cap number of evaluated folds, 0=all')
    args=p.parse_args()
    X,G,H,y,names=load_dataset(args.data_dir,args.cache,args.backend,batch=args.batch,refresh_cache=args.refresh_cache)
    args.output.mkdir(parents=True,exist_ok=True)
    foldrows=[];predrows=[]
    for seed in args.seeds:
        cv=StratifiedKFold(args.folds,shuffle=True,random_state=seed)
        for fold,(tr,te) in enumerate(cv.split(X,y)):
            if args.max_folds>0 and len(foldrows)>=args.max_folds:break
            model=CVAtlasMIL(trees=args.trees,mil_epochs=args.mil_epochs,
                    workers=args.workers,seed=seed+17*fold).fit(X[tr],G[tr],H[tr],y[tr])
            out=model.predict_scores(X[te],G[te],H[te]);pdef=out['prob_defect']
            pred=np.where(pdef>=.5,0,1)
            err=int(np.count_nonzero(pred!=y[te]))
            missed=int(np.count_nonzero((pred==1)&(y[te]==0)))
            false=int(np.count_nonzero((pred==0)&(y[te]==1)))
            foldrows.append(dict(seed=seed,fold=fold,n=len(te),errors=err,
                            missed_defects=missed,false_alarms=false,accuracy=1-err/len(te)))
            for j,i in enumerate(te):
                row=dict(seed=seed,fold=fold,image=names[i],truth=int(y[i]),
                    prediction=int(pred[j]),defect_score=float(pdef[j]))
                row.update({f'p_{k}':float(out[k][j]) for k in FUSION_WEIGHTS})
                predrows.append(row)
            _csv(args.output/'fold_metrics.csv',foldrows)
            _csv(args.output/'predictions.csv',predrows)
            print('seed',seed,'fold',fold,'errors',err,'missed',missed,'false',false,flush=True)
        if args.max_folds>0 and len(foldrows)>=args.max_folds:break
    summary={'n_predictions':len(predrows),
             'errors':sum(r['errors'] for r in foldrows),
             'missed_defects':sum(r['missed_defects'] for r in foldrows),
             'false_alarms':sum(r['false_alarms'] for r in foldrows),
             'evaluated_folds':len(foldrows),
             'full_requested_folds':len(args.seeds)*args.folds,
             'caution':'Exploratory same-dataset model selection; not independent validation'}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
