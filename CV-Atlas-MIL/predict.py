#!/usr/bin/env python3
"""Infer defective/OK labels with the trained five-expert CV Atlas MIL pipeline."""
import argparse,csv,json
from pathlib import Path
import cv2,joblib,numpy as np
from cv_atlas_mil.data import load_image,SUFFIXES
from cv_atlas_mil.features import extract_batch
from cv_atlas_mil.morphology import desc
from cv_atlas_mil import __version__

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--image',type=Path,required=True,help='One image or nonrecursive image directory')
    p.add_argument('--output',type=Path,default=Path('predictions.csv'))
    p.add_argument('--backend',choices=['auto','cuda','cython','python'],default='auto')
    p.add_argument('--batch',type=int,default=16)
    p.add_argument('--threshold',type=float,default=None)
    p.add_argument('--heatmap-dir',type=Path,default=None)
    args=p.parse_args()
    if args.threshold is not None and not 0<=args.threshold<=1:
        p.error('--threshold must be between 0 and 1')
    artifact=joblib.load(args.model)  # joblib is pickle-based: load trusted artifacts only.
    if artifact.get('version') != __version__:
        raise RuntimeError(f'Version mismatch: expected {__version__}')
    model=artifact['model']
    threshold=artifact['threshold'] if args.threshold is None else args.threshold
    if args.image.is_file():files=[args.image]
    elif args.image.is_dir():
        files=sorted(f for f in args.image.iterdir() if f.is_file() and f.suffix.lower() in SUFFIXES)
    else:raise FileNotFoundError(args.image)
    if not files:raise ValueError('No supported images found')
    if args.heatmap_dir:args.heatmap_dir.mkdir(parents=True,exist_ok=True)
    rows=[]
    for st in range(0,len(files),args.batch):
        part=files[st:st+args.batch]
        images=[load_image(f) for f in part]
        X,G=extract_batch(images,backend=args.backend)
        H=np.stack([desc(im) for im in images])
        out=model.predict_scores(X,G,H,explain=bool(args.heatmap_dir))
        for idx,path in enumerate(part):
            prob=float(out['prob_defect'][idx]);is_def=prob>=threshold
            row={'image':str(path),'prediction':0 if is_def else 1,
                 'label':'defective' if is_def else 'ok','defect_score':round(prob,6),
                 'et_score':round(float(out['base'][idx]),6),
                 'atlas_score':round(float(out['atlas'][idx]),6),
                 'mil_score':round(float(out['mil'][idx]),6),
                 'regional_score':round(float(out['regional'][idx]),6),
                 'morphology_score':round(float(out['morphology'][idx]),6),
                 'threshold':threshold}
            rows.append(row)
            if args.heatmap_dir:
                for name,key in [('mil_attention','region_attention'),
                                  ('atlas_difference','region_deviation'),
                                  ('regional_attention','joint_attention')]:
                    a=np.asarray(out[key][idx]).reshape(8,8)
                    norm=np.uint8(np.clip((a-a.min())/max(float(np.ptp(a)),1e-6)*255,0,255))
                    heat=cv2.applyColorMap(cv2.resize(norm,(512,512),interpolation=cv2.INTER_NEAREST),cv2.COLORMAP_JET)
                    overlay=cv2.addWeighted(cv2.cvtColor(images[idx],cv2.COLOR_GRAY2BGR),.65,heat,.35,0)
                    cv2.imwrite(str(args.heatmap_dir/f'{st+idx:05d}_{name}.png'),overlay)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({'count':len(rows),'output':str(args.output),
                      'preview':rows[:3]},indent=2))

if __name__=='__main__':main()
