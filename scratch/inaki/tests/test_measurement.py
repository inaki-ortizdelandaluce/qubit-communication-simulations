from qt.measurement import PVM
from qt.qubit import Qubit
import math
import numpy as np


def test_pvm_z():
    q = Qubit(np.array([1, 0]))
    z = PVM(q)
    projectors = np.array([[[1, 0],
                            [0, 0]],
                           [[0, 0],
                            [0, 1]]], dtype=np.complex_)
    assert np.allclose(z.proj, projectors)


def test_pvm_x():
    q = Qubit(1 / math.sqrt(2) * np.array([1, 1]))
    pvm = PVM(q)
    projectors = np.array([[[0.5, 0.5],
                            [0.5, 0.5]],
                           [[0.5, -0.5],
                            [-0.5, 0.5]]], dtype=np.complex_)
    assert np.allclose(pvm.proj, projectors)


def test_pvm_projector():
    q = Qubit(np.array([1, 0]))
    pvm = PVM(q)
    assert np.allclose(pvm.projector(1), np.array([[0, 0], [0, 1]]))


def test_pvm_probability():
    zero = Qubit(np.array([1, 0]))
    plus = Qubit(1 / math.sqrt(2) * np.array([1, 1]))
    minus = Qubit(1 / math.sqrt(2) * np.array([1, -1]))
    i = Qubit(1 / math.sqrt(2) * np.array([1, 1j]))
    psi = Qubit(np.array([(3 + 1.j * math.sqrt(3))/4., -0.5]))

    x = PVM(plus)
    y = PVM(i)
    z = PVM(zero)

    assert np.allclose(z.probability(plus), np.array([0.5, 0.5]))
    assert np.allclose(z.probability(minus), np.array([0.5, 0.5]))
    assert np.allclose(x.probability(minus), np.array([0., 1.]))
    assert np.allclose(x.probability(plus), np.array([1., 0.]))
    assert np.allclose(z.probability(psi), np.array([0.75, 0.25]))
    assert np.allclose(x.probability(psi), np.array([0.125, 0.875]))
    assert np.allclose(y.probability(psi), np.array([0.71650635, 0.28349365]))
