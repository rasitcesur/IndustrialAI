"""Reproducible, label-independent feature extraction for 512x512 casting images.

NOTE: The research results use the legacy `fast_kernels.pyx` operator, which is
not identical to the earlier OpenCV SIMD implementation. This module reproduces
that *Cython* definition and offers a parity-tested pure OpenCV/NumPy fallback.
"""
from __future__ import annotations
import cv2
import numpy as np
from scipy.special import expit

FEATURE_NAMES = [
    'dark_frac','dark_largest_frac','dark_n','dark_aspect','dark_top3_frac',
    'bright_frac','bright_largest_frac','bright_n','bright_aspect','bright_top3_frac',
    'highpass_mean','highpass_q95','gray_std','gray_mean',
]
G1D = cv2.getGaussianKernel(15, 0).astype(np.float32).ravel()
K0 = np.asarray([[1/16,1/8,1/16], [1/8,1/4,1/8], [1/16,1/8,1/16]], np.float32)


def _check_image(im: np.ndarray) -> np.ndarray:
    if im is None or im.shape != (512, 512) or im.dtype != np.uint8:
        raise ValueError('Expected a readable 512x512 grayscale uint8 image; no implicit resize is performed')
    return im


def base_features_python(image: np.ndarray) -> np.ndarray:
    """Cython-compatible filter math; minor float32 rounding differences possible."""
    im = _check_image(image)
    blurred = cv2.sepFilter2D(im.astype(np.float32), cv2.CV_32F, G1D, G1D,
                              borderType=cv2.BORDER_REFLECT_101)
    conv10 = np.zeros_like(blurred, dtype=np.float32)
    conv10[1:-1, 1:-1] = (3 * blurred[1:-1,1:-1] - blurred[:-2,1:-1]
        - blurred[2:,1:-1] - blurred[1:-1,:-2] - blurred[1:-1,2:])
    smoothed = cv2.filter2D(conv10, cv2.CV_32F, K0, borderType=cv2.BORDER_CONSTANT)
    neigh = np.zeros_like(conv10)
    neigh[1:-1, 1:-1] = (conv10[:-2,1:-1] + conv10[2:,1:-1]
        + conv10[1:-1,:-2] + conv10[1:-1,2:])
    yx = np.clip(np.arange(512), 1, 510)
    c = smoothed[np.ix_(yx,yx)]
    k3 = (5 * smoothed - neigh)[np.ix_(yx,yx)]
    mi = c.reshape(32,16,32,16).min(axis=(1,3))
    ma = k3.reshape(32,16,32,16).max(axis=(1,3))
    lo = expit((mi-127.5)/127.5).astype(np.float32)
    hi = np.logaddexp(0,ma/255).astype(np.float32)
    return np.concatenate((lo.ravel(), hi.ravel())).astype(np.float32)


# CUDA parity testing runs once per interpreter; absent GPU never blocks CPU mode.
_CUDA_VERIFIED = False
# Keep pinned ring buffers/streams alive between disk-read batches.
import threading
_CUDA_LOCAL = threading.local()


def available_backends():
    """Capabilities, not an assertion that numerical parity has already passed."""
    import importlib.util
    from .cuda_pipeline import cuda_available
    return {'cuda': cuda_available(),
            'cython': importlib.util.find_spec('fast_kernels') is not None,
            'python': True}


def selected_backend(requested='auto'):
    """Choose numerical-equivalence guarded CUDA -> Cython -> Python fallback."""
    global _CUDA_VERIFIED
    import warnings
    if requested not in ('auto', 'cuda', 'cython', 'python'):
        raise ValueError('backend must be auto, cuda, cython or python')
    if requested in ('auto', 'cuda'):
        from .cuda_pipeline import cuda_available, cuda_self_test
        if cuda_available():
            try:
                if not _CUDA_VERIFIED:
                    cuda_self_test()
                    _CUDA_VERIFIED = True
                return 'cuda'
            except (RuntimeError, ValueError, OSError) as exc:
                if requested == 'cuda':
                    raise RuntimeError('CUDA parity initialization failed') from exc
                warnings.warn(f'CUDA backend rejected ({exc}); using CPU fallback', RuntimeWarning)
        elif requested == 'cuda':
            raise RuntimeError('No CUDA-enabled PyTorch/NVIDIA GPU; install compatible torch CUDA wheel')
    if requested in ('auto','cython'):
        try:
            import fast_kernels
            return 'cython'
        except ImportError:
            if requested == 'cython':
                raise RuntimeError('Compile fast_kernels: python setup.py build_ext --inplace')
    return 'python'


def base_features_batch(images: np.ndarray, backend: str='auto') -> np.ndarray:
    """2048 legacy features, with strict fallback and CUDA parity guard."""
    images = np.ascontiguousarray(images, dtype=np.uint8)
    if images.ndim != 3 or images.shape[1:] != (512,512):
        raise ValueError('Expected a batch of 512x512 images')
    if len(images) == 0:
        return np.empty((0, 2048), dtype=np.float32)
    engine = selected_backend(backend)
    if engine == 'cuda':
        from .cuda_pipeline import AsyncCudaFeatureExtractor
        import os
        try:
            stream_count = int(os.environ.get('CV_ATLAS_MIL_CUDA_STREAMS', '4'))
            engine_obj = getattr(_CUDA_LOCAL, 'engine', None)
            if engine_obj is None or engine_obj.num_streams != stream_count:
                engine_obj = AsyncCudaFeatureExtractor(num_streams=stream_count)
                _CUDA_LOCAL.engine = engine_obj
            return engine_obj.process_batch(images)
        except (RuntimeError, OSError) as exc:
            if backend != 'auto':
                raise
            import warnings
            warnings.warn(f'CUDA processing failed ({exc}); reverting to CPU', RuntimeWarning)
            engine = selected_backend('cython') if available_backends()['cython'] else 'python'
    if engine == 'cython':
        import fast_kernels
        out = np.empty((len(images),2048),dtype=np.float32)
        fast_kernels.extract_features_batch(images,G1D,out)
        return out
    return np.stack([base_features_python(im) for im in images])


def reduce_to_128(features_2048: np.ndarray) -> np.ndarray:
    x = np.asarray(features_2048, np.float32)
    if x.ndim != 2 or x.shape[1] != 2048:
        raise ValueError('Expected (N,2048) original feature matrix')
    lo = x[:,:1024].reshape(-1,8,4,8,4).min(axis=(2,4)).reshape(-1,64)
    hi = x[:,1024:].reshape(-1,8,4,8,4).max(axis=(2,4)).reshape(-1,64)
    return np.ascontiguousarray(np.concatenate((lo,hi),axis=1), dtype=np.float32)


def _components(binary: np.ndarray) -> list[float]:
    contours,_ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [0.,0.,0.,0.,0.]
    areas = np.asarray([cv2.contourArea(c) for c in contours],np.float32)
    valid = areas[areas >= 2.0]
    if len(valid)==0:
        return [float(np.mean(binary != 0)),0.,0.,0.,0.]
    best = max(contours,key=cv2.contourArea)
    _,(w,h),_ = cv2.minAreaRect(best)
    aspect = max(w,h)/max(min(w,h),1.)
    return [float(np.mean(binary!=0)),float(valid.max()/4096.),
            float(min(len(valid),64)/64.),float(min(aspect,20)/20.),
            float(np.sort(valid)[-3:].sum()/4096.)]


def geometry_features(image: np.ndarray) -> np.ndarray:
    """64 tiles x 14 fixed local intensity/shape descriptors, as in CV study."""
    im = _check_image(image)
    smooth = cv2.GaussianBlur(im,(3,3),.85)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
    dark = cv2.morphologyEx(smooth,cv2.MORPH_BLACKHAT,se)
    bright = cv2.morphologyEx(smooth,cv2.MORPH_TOPHAT,se)
    db = np.uint8(dark>8)*255
    bb = np.uint8(bright>8)*255
    g3 = cv2.GaussianBlur(im,(0,0),1.1)
    g8 = cv2.GaussianBlur(im,(0,0),3.1)
    hp = cv2.absdiff(g3,g8)
    out = np.zeros((64,14),np.float32)
    for r in range(8):
        for c in range(8):
            ys=slice(r*64,(r+1)*64); xs=slice(c*64,(c+1)*64)
            p=im[ys,xs]; h=hp[ys,xs]
            out[r*8+c]=_components(db[ys,xs])+_components(bb[ys,xs])+[
                float(h.mean()/30.),float(np.percentile(h,95)/30.),
                float(p.std()/80.),float(p.mean()/255.)]
    return out


def extract_batch(images: list[np.ndarray], backend='auto') -> tuple[np.ndarray,np.ndarray]:
    if not images:
        raise ValueError('No images')
    for image in images:
        _check_image(image)
    base=reduce_to_128(base_features_batch(np.stack(images),backend))
    geom=np.stack([geometry_features(im) for im in images]).astype(np.float32)
    return base,geom
