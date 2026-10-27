"""Deterministic data discovery, feature caching, and file validation."""
from __future__ import annotations
import hashlib
from pathlib import Path
import cv2
import numpy as np
from .features import extract_batch

SUFFIXES={'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}
EXTRACTION_VERSION = 'legacy-cython-equivalent-v2-cuda-guarded'

def list_labeled(root: Path):
    root=Path(root)
    rows=[]
    for folder,label in [('def_front',0),('ok_front',1)]:
        directory=root/folder
        if not directory.is_dir():
            raise FileNotFoundError(f'Expected directory: {directory}')
        rows.extend((p,label) for p in sorted(directory.iterdir())
                    if p.is_file() and p.suffix.lower() in SUFFIXES)
    if not rows:raise ValueError(f'No supported images under {root}')
    return rows

def load_image(path:Path):
    im=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    if im is None:raise ValueError(f'Could not read {path}')
    if im.shape!=(512,512):raise ValueError(f'{path}: expected 512x512; got {im.shape}')
    return im

def _manifest(rows,root):
    h=hashlib.sha256()
    for p,y in rows:
        st=p.stat()
        h.update(f'{p.relative_to(root)}\t{y}\t{st.st_size}\t{st.st_mtime_ns}\n'.encode())
    return h.hexdigest()

def load_dataset(root:Path, cache:Path|None=None, backend='auto', batch=16, refresh_cache=False):
    root=Path(root).resolve();rows=list_labeled(root)
    names=np.asarray([str(p.relative_to(root)).replace('\\','/') for p,_ in rows])
    labels=np.asarray([y for _,y in rows],dtype=np.int32)
    signature=_manifest(rows,root)
    if cache and Path(cache).exists() and not refresh_cache:
        with np.load(cache,allow_pickle=False) as z:
            if (all(k in z for k in ('signature', 'extraction_version', 'names', 'base', 'geom'))
                and str(z['signature'])==signature
                and str(z['extraction_version'])==EXTRACTION_VERSION
                and np.array_equal(names,z['names'])
                and z['base'].shape==(len(rows),128)
                and z['geom'].shape==(len(rows),64,14)):
                print(f'Using verified feature cache: {cache} (no backend executed)',flush=True)
                return z['base'].copy(),z['geom'].copy(),labels,names
    from .features import selected_backend
    engine=selected_backend(backend)
    print(f'Feature backend: {engine} (geometry descriptors remain CPU/OpenCV)',flush=True)
    xb=[];gg=[]
    for st in range(0,len(rows),batch):
        images=[load_image(p) for p,_ in rows[st:st+batch]]
        x,g=extract_batch(images,backend=backend)
        xb.append(x);gg.append(g)
        print(f'features: {min(st+batch,len(rows))}/{len(rows)}',flush=True)
    x=np.concatenate(xb);g=np.concatenate(gg)
    if cache:
        cache=Path(cache);cache.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(cache,base=x,geom=g,y=labels,names=names,signature=signature,
                            extraction_version=EXTRACTION_VERSION)
    return x,g,labels,names


def load_dataset_v1(root:Path, cache:Path|None=None, backend='auto', batch=16,
                    refresh_cache=False):
    """Return X, G, H, y, names with cache integrity and extraction-version checks."""
    root=Path(root).resolve()
    X,G,y,names=load_dataset(root,cache,backend,batch,refresh_cache)
    rows=list_labeled(root)
    signature=_manifest(rows,root)
    morph_cache=(Path(str(cache)+'.morph.npz') if cache is not None else None)
    if morph_cache and morph_cache.exists() and not refresh_cache:
        with np.load(morph_cache,allow_pickle=False) as z:
            if (all(k in z for k in ('signature','version','names','features'))
                and str(z['signature'])==signature
                and str(z['version'])=='highres-morph-64x28-v1'
                and np.array_equal(z['names'],names)
                and z['features'].shape==(len(rows),64,28)):
                return X,G,z['features'].copy(),y,names
    from .morphology import desc
    hh=[]
    for st in range(0,len(rows),batch):
        hh.extend(desc(load_image(p)) for p,_ in rows[st:st+batch])
        print(f'morphology: {min(st+batch,len(rows))}/{len(rows)}',flush=True)
    H=np.stack(hh).astype(np.float32)
    if morph_cache:
        morph_cache.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(morph_cache,features=H,names=names,
                            signature=signature,version='highres-morph-64x28-v1')
    return X,G,H,y,names
