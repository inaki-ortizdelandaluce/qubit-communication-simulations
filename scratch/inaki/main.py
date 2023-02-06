import matplotlib.pyplot as plt
import numpy as np
from healpy.pixelfunc import ang2pix
from qt import random


def test_random_states():
    size = 100000
    nside = 4
    pixels = 12 * nside ** 2
    indexes = np.zeros(size)
    for i in range(size):
        theta, phi = random.qubit().bloch_angles()
        pix = ang2pix(nside, theta, phi)
        indexes[i] = pix

    count, bins, ignored = plt.hist(indexes, bins=range(pixels + 1), density=True)
    plt.plot(bins, np.ones_like(bins)/pixels, linewidth=2, color='r')
    plt.show()
    return None


if __name__ == "__main__":
    test_random_states()
