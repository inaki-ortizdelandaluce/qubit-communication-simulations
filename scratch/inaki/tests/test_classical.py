import numpy as np

from qt.classical import *
import qt.random as random


def test_heaviside():
    a = np.array([-1e-9, 1e-9, -1., +1.])
    assert np.allclose(np.array([0, 1, 0, 1]), heaviside(a))


def test_theta():
    a = np.array([-1e-9, 1e-9, -1., +1.])
    assert np.allclose(np.array([0, 1e-9, 0, 1]), theta(a))


def test_prepare():
    np.random.seed(0)
    lambdas = np.array([random.bloch_vector(), random.bloch_vector()])
    qubit = random.qubit()

    alice = prepare(lambdas, qubit)

    # lamda1 -> array([ 0.82760922, -0.03589592,  0.56015575])
    # lambda2 -> array([ 0.79972854, -0.38493927,  0.46071251]))
    # x -> array([-0.36250052, 0.43579601, 0.82381746])
    bits = alice['bits']

    # print('\nRandom Qubit ={}'.format(str(alice['qubit'])))

    assert (bits[0] == 1 and bits[1] == 0)


def test_measure_pvm():
    np.random.seed(0)
    lambdas = np.array([random.bloch_vector(), random.bloch_vector()])
    measurement = random.pvm()

    bob = measure_pvm(lambdas, np.array([1, 0]), measurement)

    assert np.allclose(bob['probabilities'], np.array([1, 0]))


def test_prepare_and_measure_pvm():
    np.random.seed(0)

    shots = 10**2
    experiment = prepare_and_measure_pvm(shots)
    pb1 = experiment['probabilities']['b1']
    pb2 = experiment['probabilities']['b2']

    p1 = np.sum(pb1) / len(pb1)
    p2 = np.sum(pb2) / len(pb2)

    # print('p1={},p2={},pt={}'.format(p1, p2, p1 + p2))

    assert np.allclose(experiment['probabilities']['born'], np.array([0.96687561, 0.03312439]))
    assert np.allclose(np.array([p1, p2]), np.array([0.96687561, 0.03312439]), rtol=1e-2, atol=1e-2)


def test_measure_povm():
    np.random.seed(0)
    lambdas = np.array([random.bloch_vector(), random.bloch_vector()])

    zero = np.array([[1, 0], [0, 0]])
    one = np.array([[0, 0], [0, 1]])
    plus = 0.5 * np.array([[1, 1], [1, 1]])
    minus = 0.5 * np.array([[1, -1], [-1, 1]])
    measurement = POVM(weights=0.5 * np.array([1, 1, 1, 1]), proj=np.array([zero, one, plus, minus]))

    bob = measure_povm(lambdas, np.array([1, 0]), measurement)

    assert np.allclose(bob['probabilities'], np.array([0.403638773, 0, 0.59636123, 0]))
