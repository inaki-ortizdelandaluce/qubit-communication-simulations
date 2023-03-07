import matplotlib.pyplot as plt
import numpy as np
from healpy.pixelfunc import ang2pix
import qt.classical
import qt.qubit
import qt.random
from qt.measurement import POVM


def test_random_states():
    size = 100000
    n = 4
    pixels = 12 * n ** 2
    indexes = np.zeros(size)
    for i in range(size):
        theta, phi = qt.random.qubit().bloch_angles()
        pix = ang2pix(n, theta, phi)
        indexes[i] = pix

    count, bins, ignored = plt.hist(indexes, bins=range(pixels + 1), density=True)
    plt.plot(bins, np.ones_like(bins)/pixels, linewidth=2, color='r')
    plt.show()
    return None


def test_pm_convergence():
    # run experiment
    np.random.seed(0)
    shots = 10 ** 5
    experiment = qt.classical.prepare_and_measure_pvm(shots)

    # plot probability convergence
    pb1 = experiment['probabilities']['b1']
    pb2 = experiment['probabilities']['b2']

    p1 = np.sum(pb1) / len(pb1)
    p2 = np.sum(pb2) / len(pb2)
    print('p1={},p2={},pt={}'.format(p1, p2, p1 + p2))

    p = np.cumsum(pb1) / (np.arange(len(pb1)) + 1)

    plt.plot(p)
    plt.axhline(y=p1, color='r', linestyle='-')
    plt.show()
    return None


def test_random_povm():
    np.random.seed(0)
    q1 = qt.random.qubit()
    q2 = qt.random.qubit()

    povm = POVM.new(np.array([q1, q2]))
    elements = povm.elements

    for i in range(elements.shape[0]):
        print('\nE{} eigenvalues -> {}'.format(i, np.linalg.eig(elements[i])[0]))
        print('\nE{}=\n{}'.format(i, elements[i]))
        print('E{} >=0 > -> {}'.format(i, (np.all(np.linalg.eig(elements[i])[0] >= -np.finfo(np.float32).eps))))

    # print('Sum E_i = I -> {}'.format(np.allclose(np.identity(2), np.sum(e * a[:, np.newaxis, np.newaxis], axis=0))))
    # print('Sum E_i = I -> {}'.format(np.allclose(np.identity(2), np.tensordot(_e, _a, axes=([0], [0])))))
    print('Sum E_i = I -> {}'.format(np.allclose(np.identity(2), np.sum(elements, axis=0))))

    return None


if __name__ == "__main__":
    # test_random_states()
    # test_pm_convergence()
    test_random_povm()
