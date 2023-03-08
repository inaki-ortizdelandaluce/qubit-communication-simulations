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
    p = experiment['probabilities']['p']
    stats = experiment['probabilities']['stats']
    print('p1={},p2={},pt={}'.format(stats[0], stats[1], np.sum(stats)))

    p = np.cumsum(p[:, 0]) / (np.arange(len(p[:, 0])) + 1)

    plt.plot(p)
    plt.axhline(y=stats[0], color='r', linestyle='-')
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
    test_pm_convergence()
    # test_random_povm()
