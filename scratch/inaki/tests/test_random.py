from qt.random import *


def test_pvm():
    p1, p2 = pvm()
    assert np.allclose(np.matmul(p1, p1), p1) and np.allclose(np.matmul(p2, p2), p2) \
           and np.allclose(p1 + p2, np.identity(2))
