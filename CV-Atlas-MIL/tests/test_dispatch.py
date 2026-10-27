from cv_atlas_mil.features import selected_backend, available_backends
import pytest

def test_cpu_forced():
    assert selected_backend('python') == 'python'

def test_auto_capability():
    assert selected_backend('auto') in ('cuda','cython','python')

def test_reject_bad_backend():
    with pytest.raises(ValueError):
        selected_backend('gpu')

def test_explicit_cuda_fails_clearly_when_absent():
    if not available_backends()['cuda']:
        with pytest.raises(RuntimeError):
            selected_backend('cuda')
