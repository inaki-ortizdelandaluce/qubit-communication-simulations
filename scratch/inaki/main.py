import math
import matplotlib.pyplot as plt
import numpy as np
from healpy.pixelfunc import ang2pix
from qt.prepare import random_qubit
from qiskit.visualization import plot_bloch_vector


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


def test_bloch_sphere():
    theta, phi = random_qubit().bloch_angles()
    _ = plot_bloch_vector([1., theta, phi], coord_type='spherical', title="Random state")
    plt.show()


if __name__ == "__main__":
    test_random_states()
    # test_bloch_sphere()
