import pytest
import math
import numpy as np
from qt.qubit import Qubit
from qt.measurement import POVM


@pytest.fixture
def qubit():
    q = Qubit(np.array([(3 + 1.j * math.sqrt(3)) / 4., -0.5]))
    return q


@pytest.fixture
def povm4():
    zero = np.array([[1, 0], [0, 0]])
    one = np.array([[0, 0], [0, 1]])
    plus = 0.5 * np.array([[1, 1], [1, 1]])
    minus = 0.5 * np.array([[1, -1], [-1, 1]])
    povm = POVM(weights=0.5 * np.array([1, 1, 1, 1]), proj=np.array([zero, one, plus, minus]))
    return povm


@pytest.fixture
def povm2():
    q1 = Qubit(([1, 0]))
    q2 = Qubit(([0, 1]))
    povm = POVM.new(np.array([q1, q2]))
    return povm
