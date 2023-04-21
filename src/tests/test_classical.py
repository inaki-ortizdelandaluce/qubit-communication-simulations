from qt.classical import *
from qt.qubit import X, Z
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

    shots = 10 ** 2
    experiment = prepare_and_measure_pvm(shots)

    assert np.allclose(experiment['probabilities']['born'], np.array([0.96687561, 0.03312439]))
    assert np.allclose(experiment['probabilities']['stats'], np.array([0.96687561, 0.03312439]), rtol=1e-2, atol=1e-2)


def test_measure_povm():
    np.random.seed(0)
    lambdas = np.array([random.bloch_vector(), random.bloch_vector()])

    # P4 = {1/2|0x0|, 1/2|1x1|, 1/2|+x+|, 1/2|-x-|}
    proj = np.array([[[1, 0], [0, 0]], [[0, 0], [0, 1]], [[.5, .5], [.5, .5]], [[.5, -.5], [-.5, .5]]])
    measurement = POVM(weights=0.5 * np.array([1, 1, 1, 1]), proj=proj)

    bob = measure_povm(lambdas, np.array([1, 0]), measurement)

    assert np.allclose(bob['probabilities'], np.array([0.403638773, 0, 0.59636123, 0]))


def test_prepare_and_measure_povm():
    np.random.seed(0)

    shots = 10 ** 2
    experiment = prepare_and_measure_povm(shots, 4)

    assert np.allclose(experiment['probabilities']['born'], np.array([0.0096687, 0.0057291, 0.8824570, 0.1021452]))
    assert np.allclose(experiment['probabilities']['stats'], np.array([0, 0.01, 0.91, 0.08]), rtol=1e-2, atol=1e-2)


def test_bell_singlet():
    np.random.seed(0)
    shots = 10 ** 3
    a0 = Observable(Z)
    b0 = Observable(-1 / math.sqrt(2) * (X + Z))

    alice = Qubit(a0.eigenvector(1))
    bob = (Qubit(b0.eigenvector(1)), Qubit(b0.eigenvector(-1)))

    experiment = qt.classical.bell_singlet(shots, alice, bob)
    expected = np.array([0.5 * math.cos(math.pi / 8) ** 2,
                         0.5 * math.sin(math.pi / 8) ** 2,
                         0.5 * math.sin(math.pi / 8) ** 2,
                         0.5 * math.cos(math.pi / 8) ** 2])
    assert np.allclose(experiment['probabilities']['stats'], expected, rtol=1e-1, atol=1e-1)


def test_bell_singlet_full():
    np.random.seed(0)

    shots = 10 ** 3
    a0 = Observable(Z)
    a1 = Observable(X)
    b0 = Observable(-1 / math.sqrt(2) * (X + Z))
    b1 = Observable(1 / math.sqrt(2) * (X - Z))

    experiment = bell_singlet_full(shots, alice=(a0, a1), bob=(b0, b1))

    expected = np.array([[0.5 * math.cos(math.pi / 8) ** 2,
                          0.5 * math.sin(math.pi / 8) ** 2,
                          0.5 * math.sin(math.pi / 8) ** 2,
                          0.5 * math.cos(math.pi / 8) ** 2],
                         [0.5 * math.cos(math.pi / 8) ** 2,
                          0.5 * math.sin(math.pi / 8) ** 2,
                          0.5 * math.sin(math.pi / 8) ** 2,
                          0.5 * math.cos(math.pi / 8) ** 2],
                         [0.5 * math.cos(math.pi / 8) ** 2,
                          0.5 * math.sin(math.pi / 8) ** 2,
                          0.5 * math.sin(math.pi / 8) ** 2,
                          0.5 * math.cos(math.pi / 8) ** 2],
                         [0.5 * math.cos(3 * math.pi / 8) ** 2,
                          0.5 * math.sin(3 * math.pi / 8) ** 2,
                          0.5 * math.sin(3 * math.pi / 8) ** 2,
                          0.5 * math.cos(3 * math.pi / 8) ** 2]
                         ])
    assert np.allclose(experiment['probabilities']['born'], expected)
    assert np.allclose(experiment['probabilities']['stats'], expected, rtol=1e-1, atol=1e-1)
