from qt.qubit import Qubit
from qt.qudit import Qudit
import numpy as np
import math


def test_qudit_rho():
    q = Qudit(0.5 * np.array([1, -1, 1, -1], dtype=complex))
    assert np.allclose(0.25*np.array([[1, -1, 1, -1], [-1, 1, -1, 1], [1, -1, 1, -1], [-1, 1, -1, 1]]), q.rho())


def test_qudit_bipartite():
    q1 = Qubit(1/math.sqrt(2) * np.array([1, 1]))
    q2 = Qubit(1 / math.sqrt(2) * np.array([1, -1]))
    assert np.allclose(0.5 * np.array([1, -1, 1, -1], dtype=complex), Qudit.bipartite(q1, q2).ket)
