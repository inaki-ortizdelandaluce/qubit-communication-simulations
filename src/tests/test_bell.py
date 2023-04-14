import pytest
from qt.bell import BellState, BellScenario
from qt.observable import Observable
from qt.qubit import X, Y, Z
import math


def tet_bell_scenario():
    a0 = Observable(Z)
    a1 = Observable(X)
    b0 = Observable(-1 / math.sqrt(2) * (X + Z))
    b1 = Observable(1 / math.sqrt(2) * (X - Z))
    bell = BellScenario(BellState.PSI_MINUS, alice=(a0, a1), bob=(b0, b1))
    assert True

