"""Public API and repository naming tests."""
from pathlib import Path
import pytest
from cv_atlas_mil import CVAtlasMIL, FUSION_WEIGHTS, __version__
from cv_atlas_mil.model import CVAtlasMIL as NativeCVAtlasMIL

def test_public_model_uses_native_package():
    assert CVAtlasMIL is NativeCVAtlasMIL
    assert CVAtlasMIL.__module__ == 'cv_atlas_mil.model'
    assert sum(FUSION_WEIGHTS.values()) == pytest.approx(1.0)
    assert __version__ == '1.0.0'

def test_readme_public_name():
    for name in ('README.md', 'README_TR.md'):
        title = (Path(__file__).resolve().parents[1] / name).read_text(encoding='utf8').splitlines()[0]
        assert title == '# CV Atlas MIL'
