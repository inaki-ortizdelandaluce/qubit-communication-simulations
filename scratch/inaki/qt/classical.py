import numpy as np
from qt import random as random


def heaviside(z):
    # TODO vectorise np.greater_equal
    if z >= 0:
        return 1
    else:
        return 0


def theta(z):
    # TODO vectorise np.multiply
    return z * heaviside(z)


def prepare(lambda1, lambda2):
    # TODO vectorise

    # Alice prepares and sends two qubits to Bob
    x = random.vector3()
    c1 = heaviside(np.dot(x, lambda1))
    c2 = heaviside(np.dot(x, lambda2))
    return c1, c2


def measure(lambda1, lambda2, c1, c2):
    # TODO vectorise

    # Bob receives two bits from Alice and performs the measurements
    if (c1 % 2) == 0:
        lambda1 = - lambda1

    return None


def prepare_and_measure():
    # TODO vectorise

    # shared randomness
    lambda1, lambda2 = random.vector3(), random.vector3()

    # Alice prepares
    c1, c2 = prepare(lambda1, lambda2)

    # Bob measures
    measure(lambda1, lambda2, c1, c2)
