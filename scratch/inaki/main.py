import matplotlib.pyplot as plt
import numpy as np
from healpy.pixelfunc import ang2pix
from qt.prepare import random_qubit, random_qubit_pvm


def test_random_states():
    size = 100000
    nside = 4
    pixels = 12 * nside ** 2
    indexes = np.zeros(size)
    for i in range(size):
        theta, phi = random_qubit().bloch_angles()
        pix = ang2pix(nside, theta, phi)
        indexes[i] = pix

    count, bins, ignored = plt.hist(indexes, bins=range(pixels + 1), density=True)
    plt.plot(bins, np.ones_like(bins)/pixels, linewidth=2, color='r')
    plt.show()
    return None


def test_random_pvm():
    p1, p2 = random_qubit_pvm()
    if np.allclose(p1 + p2, np.identity(2)):
        print("PVM projectors sum the identity")
    if np.allclose(np.matmul(p1, p1), p1) and np.allclose(np.matmul(p2, p2), p2):
        print("PVM projectors are idempotent")


if __name__ == "__main__":
    # test_random_states()
    test_random_pvm()
