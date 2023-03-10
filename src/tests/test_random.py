import numpy as np
import pytest

import qt.random


def test_pvm():
    # np.random.seed(0)
    pvm = qt.random.pvm()
    p1, p2 = pvm.projector(0), pvm.projector(1)

    assert np.allclose(np.matmul(p1, p1), p1) and np.allclose(np.matmul(p2, p2), p2) \
           and np.allclose(p1 + p2, np.identity(2))


def test_povm():
    # np.random.seed(0)
    povm = qt.random.povm(4)
    elements = povm.elements
    n = elements.shape[0]

    assert n == 4

    for i in range(n):
        assert (np.all(np.linalg.eig(elements[i])[0] >= -np.finfo(np.float32).eps))

    assert np.allclose(np.identity(2), np.sum(elements, axis=0))


def test_povm_raise():
    with pytest.raises(ValueError):
        qt.random.povm(2)
