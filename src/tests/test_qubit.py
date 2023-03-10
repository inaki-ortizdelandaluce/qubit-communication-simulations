from qt.qubit import *
from pytest import approx


def test_qubit_ket():
    q = Qubit(np.array([1., 1.]))
    assert np.allclose(q.ket(), 1/math.sqrt(2) * np.array([[1, 1]]))


def test_qubit_normalize():
    q = Qubit(np.array([1., 1.]))
    q.normalize()
    assert np.allclose(np.array([q.alpha, q.beta]), 1/math.sqrt(2) * np.ones((2,)))


def test_qubit_bloch_angles():
    q1 = Qubit(np.array([1., 1.j]))
    theta1, phi1 = q1.bloch_angles()

    q2 = Qubit(np.array([(1.-1.j) / (2 * math.sqrt(2)), math.sqrt(3) / 2]))
    theta2, phi2 = q2.bloch_angles()

    assert math.degrees(theta1) == approx(90.) and math.degrees(phi1) == approx(90.)
    assert math.degrees(theta2) == approx(120.) and math.degrees(phi2) == approx(45.)


def test_qubit_bloch_vector():

    q = Qubit(np.array([(1.-1.j) / (2 * math.sqrt(2)), math.sqrt(3) / 2]))
    xyz = q.bloch_vector()

    assert np.allclose(np.asarray(xyz), np.array([0.61237, 0.61237, -0.499999]))


def test_qubit_rho():
    q = Qubit(np.array([1., -1.]))
    assert np.allclose(q.rho(), 0.5 * np.array([[1., -1.], [-1., 1.]]))
