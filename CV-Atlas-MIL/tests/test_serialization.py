"""Native package path is preserved in new trained model artifacts."""
import joblib
import numpy as np
from cv_atlas_mil import CVAtlasMIL


def test_native_model_roundtrip(tmp_path):
    rng = np.random.default_rng(52)
    n = 12
    X = rng.random((n, 128)).astype(np.float32)
    G = rng.random((n, 64, 14)).astype(np.float32)
    H = rng.random((n, 64, 28)).astype(np.float32)
    y = np.array([0, 1] * (n // 2), dtype=int)
    before = CVAtlasMIL(trees=6, mil_epochs=1, workers=1, torch_threads=1).fit(X, G, H, y)
    scores_before = before.predict_scores(X[:2], G[:2], H[:2])
    path = tmp_path / 'cv_atlas_mil.joblib'
    joblib.dump({'version': '1.0.0', 'model': before}, path)
    after = joblib.load(path)['model']
    assert type(after).__module__ == 'cv_atlas_mil.model'
    assert type(after.original).__module__ == 'cv_atlas_mil.pipeline'
    assert type(after.original.atlas).__module__ == 'cv_atlas_mil.atlas'
    scores_after = after.predict_scores(X[:2], G[:2], H[:2])
    for key in ('prob_defect', 'base', 'atlas', 'mil', 'regional', 'morphology'):
        np.testing.assert_allclose(scores_after[key], scores_before[key], rtol=0, atol=0)
