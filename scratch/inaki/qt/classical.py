import math
import numpy as np
from qt import random as random


def heaviside(a):
    return (a >= 0).astype(int)


def theta(a):
    return a * heaviside(a)


def prepare(lambda1, lambda2):
    """
    Alice prepares and sends two bits to Bob

    :param lambda1: first shared random normalized vector
    :param lambda2: second shared random normalized vector
    :return: the two bits to be communicated to Bob
    """
    #
    x = random.vector3()
    lambdas = np.array([lambda1, lambda2])
    bits = heaviside(np.multiply(x, lambdas.T))
    return bits[0], bits[1]


def measure_pvm(lambda1, lambda2, bit1, bit2):
    """
    Bob receives two bits from Alice and performs a random PVM

    :param lambda1: first shared random normalized vector
    :param lambda2: second shared random normalized vector
    :param bit1: first bit communicated by Alice
    :param bit2: second bit communicated by Alice
    :return: the probabilities for the random measurement
    """

    lambdas = np.array([lambda1, lambda2])
    bits = np.array([[bit1], [bit2]])

    # flip shared randomness
    flip = np.where(bits == 0, -1, 1)
    lambdas = lambdas * flip

    # generate classical random PVM as vectors
    y = np.asarray(random.pvm_vectors())

    # select lambdas for each measurement
    a = np.abs(np.mathmul(lambdas, y.T))
    lambdas = lambdas[np.argmax(a, axis=0), :]

    # compute probabilities
    thetas = theta(np.multiply(y, lambdas.T))

    p = np.diag(thetas) / np.sum(thetas, axis=0)

    return p


def prepare_and_measure():

    # Alice and Bob's shared randomness
    lambda1, lambda2 = random.vector3(), random.vector3()

    # Alice prepares
    bit1, bit2 = prepare(lambda1, lambda2)

    # Bob measures
    measure_pvm(lambda1, lambda2, bit1, bit2)
