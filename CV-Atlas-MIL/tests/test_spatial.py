"""Cross-family region alignment, mixture weights and train-only model smoke tests."""
import numpy as np
import pytest
from cv_atlas_mil.regional import fit_normalizer,co_located_vectors
from cv_atlas_mil.model import CVAtlasMIL,FUSION_WEIGHTS
from cv_atlas_mil.morphology import desc


def test_regional_vectors_preserve_exact_region_index():
    n=3
    g=np.zeros((n,64,14),dtype=np.float32)
    h=np.zeros((n,64,28),dtype=np.float32)
    a=np.zeros((n,64,4),dtype=np.float32)
    for r in range(64):
        g[:,r,0]=r*.1
        h[:,r,0]=r*.2
        a[:,r,2]=r*.3
    gn=(np.zeros((64,14),dtype=np.float32),np.ones((64,14),dtype=np.float32))
    hn=(np.zeros((64,28),dtype=np.float32),np.ones((64,28),dtype=np.float32))
    z=co_located_vectors(g,h,a,gn,hn)
    assert z.shape==(n,64,48)
    assert z[0,11,0]==pytest.approx(1.1)
    assert z[0,11,14]==pytest.approx(2.2)
    assert z[0,11,44]==pytest.approx(np.log1p(3.3),rel=1e-6)
    assert z[0,12,0]!=z[0,11,0]


def test_morphology_layout():
    im=np.random.default_rng(8).integers(0,255,size=(512,512),dtype=np.uint8)
    H=desc(im)
    assert H.shape==(64,28)
    assert np.isfinite(H).all()


def test_fusion_weights_are_valid():
    assert sum(FUSION_WEIGHTS.values())==pytest.approx(1.0)
    assert all(w>=0 for w in FUSION_WEIGHTS.values())


def test_full_model_fit_and_predict_from_synthetic_feature_arrays():
    rng=np.random.default_rng(49)
    n=20
    g=rng.normal(0.3,.1,(n,64,14)).astype(np.float32)
    x=rng.uniform(.25,.75,(n,128)).astype(np.float32)
    h=rng.uniform(0,.3,(n,64,28)).astype(np.float32)
    y=np.array([0]*10+[1]*10)
    model=CVAtlasMIL(trees=8,mil_epochs=1,workers=1,torch_threads=1).fit(x,g,h,y)
    p=model.predict_scores(x[:3],g[:3],h[:3],explain=True)
    assert p['prob_defect'].shape==(3,)
    assert np.isfinite(p['prob_defect']).all()
    np.testing.assert_allclose(p['prob_defect'],sum(FUSION_WEIGHTS[k]*p[k] for k in FUSION_WEIGHTS),rtol=1e-6,atol=1e-6)
    assert p['joint_attention'].shape==(3,64)
    assert (model.predict(x[:3],g[:3],h[:3])==np.where(p['prob_defect']>=.5,0,1)).all()
