import numpy as np
import qt.random


def test_pvm():
    pvm = qt.random.pvm()
    p1, p2 = pvm.projector(0), pvm.projector(1)

    assert np.allclose(np.matmul(p1, p1), p1) and np.allclose(np.matmul(p2, p2), p2) \
           and np.allclose(p1 + p2, np.identity(2))

