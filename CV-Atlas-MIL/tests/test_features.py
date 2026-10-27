"""Integration tests. Native test is optional; run after building extension."""
import numpy as np
from cv_atlas_mil.features import base_features_python,extract_batch,base_features_batch

def test_shapes():
    a=np.random.default_rng(27).integers(0,256,(512,512),dtype=np.uint8)
    X,G=extract_batch([a],backend='python')
    assert X.shape==(1,128) and G.shape==(1,64,14)
    assert np.isfinite(X).all() and np.isfinite(G).all()

def test_native_parity():
    try:import fast_kernels
    except ImportError:return
    images=np.random.default_rng(8).integers(0,256,(2,512,512),dtype=np.uint8)
    a=base_features_batch(images,'cython')
    b=base_features_batch(images,'python')
    np.testing.assert_allclose(a,b,rtol=1e-3,atol=2e-3)
