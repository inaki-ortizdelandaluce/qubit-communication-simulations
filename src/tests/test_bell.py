import pytest
from qt.bell import BellState, BellScenario
from qt.observable import Observable
from qt.qubit import X, Y, Z
import numpy as np
import math


def tet_bell_scenario_chsh():
    a1 = Observable(X)
    a0 = Observable(Z)
    b0 = Observable(-1 / math.sqrt(2) * (X + Z))
    b1 = Observable(1 / math.sqrt(2) * (X - Z))

    bell = BellScenario(BellState.PSI_MINUS, alice=(a0, a1), bob=(b0, b1))
    assert np.isclose(2 * math.sqrt(2), bell.chsh())


def tet_bell_scenario_probability():
    a0 = Observable(Z)
    a1 = Observable(X)
    b0 = Observable(-1 / math.sqrt(2) * (X + Z))
    b1 = Observable(1 / math.sqrt(2) * (X - Z))

    bell = BellScenario(BellState.PSI_MINUS, alice=(a0, a1), bob=(b0, b1))
    assert np.isclose(0.5 * math.cos(math.pi / 8) ** 2, bell.probability()[0, 0]) \
           and np.isclose(0.5 * math.sin(math.pi / 8) ** 2, bell.probability()[0, 1])


def tet_bell_scenario_expectation_values():
    a0 = Observable(Z)
    a1 = Observable(X)
    b0 = Observable(-1 / math.sqrt(2) * (X + Z))
    b1 = Observable(1 / math.sqrt(2) * (X - Z))

    bell = BellScenario(BellState.PSI_MINUS, alice=(a0, a1), bob=(b0, b1))
    assert np.allclose(0.5 * np.array([math.cos(math.pi / 8) ** 2,
                                       math.cos(math.pi / 8) ** 2,
                                       math.cos(math.pi / 8) ** 2,
                                       math.cos(3 * math.pi / 8) ** 2]), bell.expected_values())
