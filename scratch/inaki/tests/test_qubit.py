import numpy as np

from qt.qubit import *
from pytest import approx


def test_qubit_to_array():
    q = Qubit(1., 1.)
    assert np.allclose(q.to_array(), 1/math.sqrt(2) * np.array([[1, 1]]))


def test_qubit_normalize():
    q = Qubit(1., 1.)
    q.normalize()
    assert np.allclose(np.array([q.alpha, q.beta]), 1/math.sqrt(2) * np.ones((2,)))


def test_qubit_bloch_angles():
    q1 = Qubit(1., 1.j)
    theta1, phi1 = q1.bloch_angles()

    q2 = Qubit((1.-1.j) / (2 * math.sqrt(2)), math.sqrt(3) / 2)
    theta2, phi2 = q2.bloch_angles()

    assert math.degrees(theta1) == approx(90.) and math.degrees(phi1) == approx(90.)
    assert math.degrees(theta2) == approx(120.) and math.degrees(phi2) == approx(45.)


def test_qubit_to_density_matrix():
    q = Qubit(1., -1.)
    assert np.allclose(q.to_density_matrix(), 0.5 * np.array([[1., -1.], [-1., 1.]]))
