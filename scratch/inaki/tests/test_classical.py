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
    assert True

