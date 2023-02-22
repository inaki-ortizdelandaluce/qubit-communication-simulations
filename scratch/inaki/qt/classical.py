import numpy as np
from qt import random as random


def heaviside(a):
    if isinstance(a, np.ndarray):
        return (a >= 0).astype(int)
    else:
        return int(a >= 0)


def theta(a):
    return a * heaviside(a)


def prepare(lambdas):
    """
    Alice prepares and sends two bits to Bob

    :param lambdas: shared randomness as normalized vectors in a numpy 1-d array
    :return: a dictionary with the shared randomness ('lambdas'), the random qubit prepared by Alice ('qubit')
     and the bits to be communicated to Bob ('bits')
    """

    q = random.qubit()
    x = q.bloch_vector()
    bits = heaviside(np.matmul(x, lambdas.T))
    return {
        "lambdas": lambdas,
        "qubit": q,
        "bits": bits
    }


def measure_pvm(lambdas, bits):
    """
    Bob receives two bits from Alice and performs a random PVM

    :param lambdas: shared randomness as normalized vectors in a numpy 1-d array
    :param bits: bits communicated by Alice in a numpy 1-d array
    :return: a dictionary with the random measurement ('measurement') and
        the probabilities for each measurement outcome ('probabilities')
    """
    bits = bits.reshape((bits.size, 1))

    # flip shared randomness
    flip = np.where(bits == 0, -1, 1)
    lambdas = np.multiply(lambdas, flip)

    # generate classical random PVM as vectors
    y = np.asarray(random.pvm_vectors())

    # select lambdas for each measurement
    a = np.abs(np.matmul(lambdas, y.T))
    lambdas = lambdas[np.argmax(a, axis=0), :]

    # compute probabilities
    thetas = theta(np.matmul(y, lambdas.T))

    p = np.diag(thetas) / np.sum(thetas, axis=0)

    return {
        "measurement": y,
        "probabilities": p
    }


def prepare_and_measure():

    # Alice and Bob's shared randomness
    shared_randomness = np.array([random.vector3(), random.vector3()])

    # Alice prepares
    alice = prepare(shared_randomness)

    # Bob measures
    bob = measure_pvm(shared_randomness, alice['bits'])
