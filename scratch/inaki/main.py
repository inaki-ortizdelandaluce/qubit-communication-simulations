import bisect

import matplotlib.pyplot as plt
import math
import numpy as np
import scipy
from healpy.pixelfunc import ang2pix
import qt.classical
import qt.qubit
import qt.random


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

    q1 = qt.random.qubit()
    q2 = qt.random.qubit()

    e3 = np.identity(2) - q1.rho() - q2.rho()
    _, w = np.linalg.eig(e3)
    q3 = qt.qubit.Qubit(w[:, 0])
    q4 = qt.qubit.Qubit(w[:, 1])

    qubits = np.array([q1, q2, q3, q4])
    v = np.asarray([q.bloch_vector() for q in qubits])

    a = np.vstack((np.ones((4,)), v.T))
    b = np.array([2, 0, 0, 0])
    lp = scipy.optimize.linprog(np.ones(4, ), A_eq=a, b_eq=b, bounds=(0.01, 1), method='highs')

    eps = np.finfo(np.float32).eps

    _a, _e = lp['x'], np.asarray([q.rho() for q in qubits])
    elements = _e * _a[:, np.newaxis, np.newaxis]
    for i in range(elements.shape[0]):
        # print('\nE{}=\n{}'.format(i, elements[i]))
        # print('\nE{} eigenvalues -> {}'.format(i, np.linalg.eig(elements[i])[0]))
        print('E{} >=0 > -> {}'.format(i, (np.all(np.linalg.eig(elements[i])[0] >= -eps))))

    # print('Sum E_i = I -> {}'.format(np.allclose(np.identity(2), np.sum(e * a[:, np.newaxis, np.newaxis], axis=0))))
    print('Sum E_i = I -> {}'.format(np.allclose(np.identity(2), np.tensordot(_e, _a, axes=([0], [0])))))

    return None


if __name__ == "__main__":
    # test_random_states()
    # test_pm_convergence()
    test_random_povm()
