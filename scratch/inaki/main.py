import matplotlib.pyplot as plt
import numpy as np
from healpy.pixelfunc import ang2pix
import qt.random
import qt.classical


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
    experiment = qt.classical.prepare_and_measure(shots)

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


if __name__ == "__main__":
    # test_random_states()
    test_pm_convergence()
