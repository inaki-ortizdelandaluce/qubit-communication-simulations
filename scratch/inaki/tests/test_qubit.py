import numpy as np
from qt.qubit import *


def test_qubit_to_array():
    q = Qubit(1, 1)
    assert np.allclose(q.to_array(), 1/math.sqrt(2) * np.array([[1, 1]]))
