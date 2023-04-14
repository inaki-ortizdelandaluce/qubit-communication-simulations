from qt.observable import Observable
import pytest
import numpy as np
import math


def test_observable_raise_error():
    with pytest.raises(ValueError, match='Input matrix is not hermitian'):
        Observable(np.array([[1, 1 + 1.j], [-1 + 1.j, 1]]))


def test_observable_eigenvalues():
    y = Observable(np.array([[0, -1.j], [1.j, 0]]))
    assert np.allclose(y.eigenvalues, np.array([1., -1.]))


def test_observable_eigenvectors():
    x = Observable(np.array([[0, 1], [1, 0]]))
    assert np.allclose(x.eigenvector(1), 1/math.sqrt(2) * np.array([1., 1.]))
    assert np.allclose(x.eigenvector(-1), 1 / math.sqrt(2) * np.array([-1., 1.]))
    assert x.eigenvector(0) is None
