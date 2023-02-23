import numpy as np
import qt.random as random
from qt.qubit import Qubit


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
    # flip shared randomness
    flip = np.where(bits == 0, -1, 1).reshape(2, 1)
    lambdas = np.multiply(lambdas, flip)

    # generate classical random PVM as Bloch vectors
    pvm = random.pvm()
    y = np.array([Qubit(pvm.basis[0]).bloch_vector(),
                  Qubit(pvm.basis[1]).bloch_vector()])

    # select lambdas for each measurement
    a = np.abs(np.matmul(lambdas, y.T))
    lambdas = lambdas[np.argmax(a, axis=0), :]

    # compute probabilities
    thetas = theta(np.matmul(y, lambdas.T))

    print('\nThetas=\n{}'.format(thetas))

    p = np.diag(thetas) / np.sum(thetas, axis=0)

    return {
        "measurement": pvm,
        "probabilities": p
    }


def prepare_and_measure():

    # Alice and Bob's shared randomness
    shared_randomness = np.array([random.bloch_vector(), random.bloch_vector()])

    # Alice prepares
    alice = prepare(shared_randomness)

    # Bob measures
    bob = measure_pvm(shared_randomness, alice['bits'])

    print('Shared randomness=\n{}'.format(alice['lambdas']))
    print('Random state=\n{}'.format(alice['qubit']))
    print('Random PVM=\n\tBasis:\n\t{}\n\tProjector:\n\t{}'.format(bob['measurement'].basis, bob['measurement'].proj))
    print('Simulation Probabilities=\n{}'.format(bob['probabilities']))
    print('Born\'s Rule Probabilities=\n{}'.format(bob['measurement'].probability(alice['qubit'].rho())))

    return {**alice, **bob}

