from qt.measurement import PVM
import math
import numpy as np


def test_measurement_pvm_z():
    z = PVM()
    projectors = np.array([[[1, 0],
                            [0, 0]],
                           [[0, 0],
                            [0, 1]]], dtype=np.complex_)
    assert np.allclose(z.proj, projectors)


def test_measurement_pvm_x():
    x = PVM(np.array([[1 / math.sqrt(2), 1 / math.sqrt(2)], [1 / math.sqrt(2), -1 / math.sqrt(2)]]))
    projectors = np.array([[[0.5, 0.5],
                            [0.5, 0.5]],
                           [[0.5, -0.5],
                            [-0.5, 0.5]]], dtype=np.complex_)
    assert np.allclose(x.proj, projectors)


def test_measurement_pvm_projector():
    z = PVM()
    assert np.allclose(z.projector(1), np.array([[0, 0], [0, 1]]))


def test_measurement_pvm_probability():
    plus = np.array([1 / math.sqrt(2), 1 / math.sqrt(2)])
    minus = np.array([1 / math.sqrt(2), -1 / math.sqrt(2)])
    psi = np.array([(3 + 1.j * math.sqrt(3))/4., -0.5])

    rho_plus = np.outer(plus, plus.conj())
    rho_minus = np.outer(minus, minus.conj())
    rho_psi = np.outer(psi, psi.conj())

    x = PVM(np.array([[1 / math.sqrt(2), 1 / math.sqrt(2)], [1 / math.sqrt(2), -1 / math.sqrt(2)]]))
    y = PVM(np.array([[1 / math.sqrt(2), 1.j / math.sqrt(2)], [1 / math.sqrt(2), -1.j / math.sqrt(2)]]))
    z = PVM()

    assert np.allclose(z.probability(rho_plus), np.array([0.5, 0.5])) \
           and np.allclose(z.probability(rho_minus), np.array([0.5, 0.5])) \
           and np.allclose(x.probability(rho_minus), np.array([0., 1.])) \
           and np.allclose(x.probability(rho_plus), np.array([1., 0.])) \
           and np.allclose(z.probability(rho_psi), np.array([0.75, 0.25])) \
           and np.allclose(x.probability(rho_psi), np.array([0.125, 0.875])) \
           and np.allclose(y.probability(rho_psi), np.array([0.71650635, 0.28349365]))
