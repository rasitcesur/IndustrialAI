"""Math parity tests do not require NVIDIA GPU; actual CUDA test skips if absent."""
import cv2
import numpy as np
import pytest
import torch
from scipy.special import expit
from cv_atlas_mil.cuda_pipeline import _torch_raw_maps, cuda_available, cuda_self_test
from cv_atlas_mil.features import base_features_python, base_features_batch, reduce_to_128


def test_cuda_graph_cpu_math_exact():
    gaussian = cv2.getGaussianKernel(15, 0).astype(np.float32).ravel()
    as_filter = lambda k: torch.from_numpy(np.array(k,dtype=np.float32,copy=True).reshape(1,1,*np.shape(k)))
    g = {'horizontal': as_filter(gaussian.reshape(1,15)),
         'vertical': as_filter(gaussian.reshape(15,1))}
    k0 = as_filter([[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]])
    k1 = as_filter([[0,-1,0], [-1,3,-1], [0,-1,0]])
    neighbors = as_filter([[0,1,0], [1,0,1], [0,1,0]])
    samples = [np.random.default_rng(25).integers(0, 256, (512,512),dtype=np.uint8),
               np.where(np.indices((512,512))[1] > 200, 255, 0).astype(np.uint8)]
    for a in samples:
        low,high = _torch_raw_maps(torch.tensor(a.copy()).float()[None,None], g,k0,k1,neighbors)
        a_raw = low.detach().numpy().reshape(-1)
        b_raw = high.detach().numpy().reshape(-1)
        got = np.r_[expit((a_raw-127.5)/127.5),np.logaddexp(0,b_raw/255)]
        np.testing.assert_allclose(got,base_features_python(a),rtol=1e-5,atol=2e-5)


def test_cpu_backend_is_deterministic():
    a=np.random.default_rng(12).integers(0,256,(1,512,512),dtype=np.uint8)
    first=base_features_batch(a,backend='python')
    second=base_features_batch(a,backend='python')
    assert first.shape==(1,2048)
    np.testing.assert_array_equal(first,second)
    assert reduce_to_128(first).shape==(1,128)


@pytest.mark.skipif(not cuda_available(), reason='CUDA-capable PyTorch and NVIDIA GPU not available')
def test_cuda_pinned_stream_parity():
    maximum=cuda_self_test(tolerance=3e-4)
    assert maximum<=3e-4
    a=np.random.default_rng(37).integers(0,256,(7,512,512),dtype=np.uint8)
    got=base_features_batch(a,backend='cuda')
    expected=base_features_batch(a,backend='python')
    np.testing.assert_allclose(got,expected,rtol=5e-4,atol=3e-4)
