import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from healpy.pixelfunc import ang2pix
from qt.__random__ import random_qubit


def test_plot_distribution():
    # x = np.random.normal(0, 1, size=1000)
    # plt.hist(x, bins=50)
    # plt.gca().set(title='Frequency Histogram', ylabel='Frequency')
    # plt.show()

    # x, _ = np.histogram([1, 2, 1, 3, 2, 1], bins=[0, 1, 2, 3], density='True')
    # plt.hist(x, bins='auto')
    # plt.show()

    # x = np.array([22, 87, 5, 43, 56, 73, 55, 54, 11, 20, 51, 5, 79, 31, 27])
    # plt.hist(x, bins=[0, 20, 40, 60, 80, 100])
    # plt.show()

    # x = np.array([22, 87, 5, 43, 56, 73, 55, 54, 11, 20, 51, 5, 79, 31, 27])
    # h, _ = np.histogram(x, bins=x, density='False')
    # plt.hist(h, bins='auto')
    # plt.show()

    # x = np.array([22, 87, 5, 43, 56, 73, 55, 54, 11, 20, 51, 5, 79, 31, 27])
    # h, _ = np.histogram(x, bins=np.arange(x.size), density='True')
    # plt.hist(h, bins='auto')
    # plt.show()

    # mu, sigma = 0, 0.1
    # s = np.random.normal(mu, sigma, 1000)
    # count, bins, ignored = plt.hist(s, 30, density=True)
    # plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) *
    #          np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)),
    #          linewidth=2, color='r')
    # plt.show()

    s = np.random.uniform(-1, 0, 1000)
    count, bins, ignored = plt.hist(s, 15, density=True)
    plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
    plt.show()

    '''
    mean = 170
    data = np.random.normal(170, 10, 250)

    # Fit a normal distribution to
    # the data:
    # mean and standard deviation
    mu, std = norm.fit(data)

    # Plot the histogram.
    plt.hist(data, bins=25, density=True, alpha=0.6, color='b')

    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)

    plt.plot(x, p, 'k', linewidth=2)
    title = "Fit Values: {:.2f} and {:.2f}".format(mu, std)
    plt.title(title)

    plt.show()
    '''


def test_random_states():
    size = 100000
    nside = 4
    pixels = 12 * nside **2
    indexes = np.zeros(size)
    for i in range(size):
        theta, phi = random_qubit().bloch_angles()
        pix = ang2pix(nside, theta, phi)
        indexes[i] = pix

    count, bins, ignored = plt.hist(indexes, bins=range(pixels + 1), density=True)
    plt.plot(bins, np.ones_like(bins)/pixels, linewidth=2, color='r')
    plt.show()
    '''
    nside = 2
    size = 180*360
    indexes = np.zeros(size)
    counter = 0
    for theta in np.arange(0, 180, 1):
        for phi in np.arange(0, 360, 1):
            print('Theta:{}'.format(str(theta)))
            pix = ang2pix(nside, math.radians(theta), math.radians(phi), nest=False)
            indexes[counter] = pix
            counter += 1
    '''

    '''
    hist, _ = np.histogram(indexes, bins=np.arange(12 * nside**2), density=False)
    count, bins, ignored = plt.hist(hist, bins='auto', density=False)
    # plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
    plt.show()
    '''
    return None


if __name__ == "__main__":
    print("Hello Quantum World")
    test_random_states()

