from qt.measurement import PVM, POVM
from qt.qubit import Qubit
import math
import numpy as np


def test_pvm_z():
    q = Qubit(np.array([1, 0]))
    z = PVM.new(q)
    projectors = np.array([[[1, 0],
                            [0, 0]],
                           [[0, 0],
                            [0, 1]]], dtype=np.complex_)
    assert np.allclose(z.proj, projectors)


def test_pvm_x():
    q = Qubit(1 / math.sqrt(2) * np.array([1, 1]))
    pvm = PVM.new(q)
    projectors = np.array([[[0.5, 0.5],
                            [0.5, 0.5]],
                           [[0.5, -0.5],
                            [-0.5, 0.5]]], dtype=np.complex_)
    assert np.allclose(pvm.proj, projectors)


def test_pvm_projector():
    q = Qubit(np.array([1, 0]))
    pvm = PVM.new(q)
    assert np.allclose(pvm.projector(1), np.array([[0, 0], [0, 1]]))


def test_pvm_probability():
    zero = Qubit(np.array([1, 0]))
    plus = Qubit(1 / math.sqrt(2) * np.array([1, 1]))
    minus = Qubit(1 / math.sqrt(2) * np.array([1, -1]))
    i = Qubit(1 / math.sqrt(2) * np.array([1, 1j]))
    psi = Qubit(np.array([(3 + 1.j * math.sqrt(3))/4., -0.5]))

    x = PVM.new(plus)
    y = PVM.new(i)
    z = PVM.new(zero)

    assert np.allclose(z.probability(plus), np.array([0.5, 0.5]))
    assert np.allclose(z.probability(minus), np.array([0.5, 0.5]))
    assert np.allclose(x.probability(minus), np.array([0., 1.]))
    assert np.allclose(x.probability(plus), np.array([1., 0.]))
    assert np.allclose(z.probability(psi), np.array([0.75, 0.25]))
    assert np.allclose(x.probability(psi), np.array([0.125, 0.875]))
    assert np.allclose(y.probability(psi), np.array([0.71650635, 0.28349365]))


def test_povm_init(povm4):
    assert np.allclose(np.identity(2), np.sum(povm4.elements, axis=0))
    assert np.allclose(np.array([[0, 0, 1], [0, 0, -1], [1, 0, 0], [-1, 0, 0]]), povm4.bloch)


def test_povm_new(povm2):
    elements = np.array([[[0.01, 0], [0, 0]], [[0, 0], [0, 0.01]], [[0.99, 0], [0, 0]], [[0, 0], [0, 0.99]]],
                        dtype=np.complex_)
    assert np.allclose(povm2.elements, elements)


def test_povm_new_probability(qubit):
    zero = Qubit(([1, 0]))
    one = Qubit(([0, 1]))
    povm = POVM.new(np.array([zero, one]))

    assert np.allclose(povm.probability(qubit), np.array([0.0075, 0.0025, 0.7424999, 0.2475]))


def test_povm_init_probability(qubit, povm4):
    assert np.allclose(np.array([0.375, 0.125, 0.0625, 0.4375]), povm4.probability(qubit))


def test_povm_len(povm4):
    assert povm4.size() == 4


def test_povm_unitary(povm4):
    u1 = povm4.unitary()
    u2 = np.array([[0.70710678 + 0.j, 0. + 0.j, 0.70710678 + 0.j, 0. + 0.j],
                  [0. + 0.j, 0.70710678 + 0.j, 0. + 0.j, 0.70710678 + 0.j],
                  [0.5 + 0.j, -0.5 + 0.j, -0.5 + 0.j, 0.5 + 0.j],
                  [0.5 + 0.j, 0.5 + 0.j, -0.5 + 0.j, -0.5 + 0.j]])
    assert np.allclose(u1, u2)
