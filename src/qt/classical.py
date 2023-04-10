import numpy as np
import random
import qt.random
from qt.qubit import Qubit
from qt.measurement import PVM, POVM


def heaviside(a):
    if isinstance(a, np.ndarray):
        return (a >= 0).astype(int)
    else:
        return int(a >= 0)


def theta(a):
    return a * heaviside(a)


def prepare(lambdas, qubit):
    """
    Alice prepares and sends two bits to Bob

    Parameters
    ---------
    lambdas : ndarray
        Shared randomness as two normalized vectors in a numpy 2-d array

    qubit: Qubit
        A uniformly sampled pure state qubit

    Returns
    -------
    dict
        A dictionary with the shared randomness ('lambdas'), the random qubit ('qubit')
        and the bits to be communicated to Bob ('bits')
    """

    x = qubit.bloch_vector()
    bits = heaviside(np.matmul(x, lambdas.T))
    return {
        "lambdas": lambdas,
        "qubit": qubit,
        "bits": bits
    }


def measure_pvm(lambdas, bits, measurement: PVM):
    """
    Bob receives two bits from Alice and performs a random PVM

    Parameters
    ---------
    lambdas : ndarray
        Shared randomness as two normalized vectors in a numpy 2-d array

    bits: ndarray
        Bits communicated by Alice in a numpy 1-d array

    measurement: PVM
        A uniformly sampled PVM

    Returns
    -------
    dict
        A dictionary with the random measurement ('measurement') and
        the probabilities for each measurement outcome ('probabilities')
    """
    # flip shared randomness
    flip = np.where(bits == 0, -1, 1).reshape(2, 1)
    lambdas = np.multiply(lambdas, flip)

    # pick bloch vectors from PVM elements
    y = measurement.bloch
    pb = 0.5 * np.array([1, 1])

    # compute lambda according to the probabilities {pb}
    index = random.choices(range(0, 2), cum_weights=np.cumsum(pb), k=1)[0]
    a = np.abs(np.matmul(lambdas, y.T))
    _lambda = lambdas[np.argmax(a, axis=0)[index]]

    # compute probabilities
    thetas = theta(np.matmul(y, _lambda.reshape(-1, 1)))
    weighted_thetas = np.multiply(thetas, pb.reshape(-1, 1))
    p = weighted_thetas[:, 0] / np.sum(weighted_thetas, axis=0)

    return {
        "measurement": measurement,
        "probabilities": p
    }


def prepare_and_measure_pvm(shots):
    """
    Runs a prepare-and-measure classical simulation with a random PVM measurement

    Parameters
    ---------
    shots : int
        Number of shots the simulation is run with

    Returns
    -------
    dict
        A dictionary with the random state ('qubit'), random PVM measurement ('measurement') and
        the probabilities for each measurement outcome ('probabilities') in a nested structure including the
        theoretical probability ('born'), the execution runs ('runs') and the probability statistics ('stats')
    """

    # Alice prepares a random qubit
    qubit = qt.random.qubit()

    # Bob prepares a random measurement
    measurement = qt.random.pvm()

    experiment = {
        "qubit": qubit,
        "measurement": measurement,
        "probabilities": {
            "runs": np.zeros((shots, 2)),
            "stats": np.zeros((2,)),
            "born": np.ones((2,))
        }
    }

    for i in range(shots):

        # Alice and Bob's shared randomness
        shared_randomness = np.array([qt.random.bloch_vector(), qt.random.bloch_vector()])

        # Alice prepares
        alice = prepare(shared_randomness, qubit)

        # Bob measures
        bob = measure_pvm(shared_randomness, alice['bits'], measurement)

        # save simulation runs
        p = np.abs(bob['probabilities'])
        experiment['probabilities']['runs'][i, :] = p

        # accumulate counts according to Bob's probabilities
        index = random.choices(range(0, 2), cum_weights=np.cumsum(p), k=1)[0]
        experiment['probabilities']['stats'][index] = experiment['probabilities']['stats'][index] + 1

    experiment['probabilities']['stats'] = experiment['probabilities']['stats'] / shots
    experiment['probabilities']['born'] = bob['measurement'].probability(qubit)

    return experiment


def measure_povm(lambdas, bits, measurement: POVM):
    """
    Bob receives two bits from Alice and performs a random POVM

    Parameters
    ---------
    lambdas : ndarray
        Shared randomness as two normalized vectors in a numpy 2-d array

    bits: ndarray
        Bits communicated by Alice in a numpy 1-d array

    measurement: POVM
        A uniformly sampled POVM

    Returns
    -------
    dict
        A dictionary with the random measurement ('measurement') and
        the probabilities for each measurement outcome ('probabilities')
    """
    # flip shared randomness
    flip = np.where(bits == 0, -1, 1).reshape(2, 1)
    lambdas = np.multiply(lambdas, flip)

    # pick bloch vectors and weights from POVM elements
    y = measurement.bloch
    pb = measurement.weights / 2.

    # compute lambda according to the probabilities {pb}
    index = random.choices(range(0, measurement.size()), cum_weights=np.cumsum(pb), k=1)[0]
    a = np.abs(np.matmul(lambdas, y.T))
    _lambda = lambdas[np.argmax(a, axis=0)[index]]

    # compute probabilities
    thetas = theta(np.matmul(y, _lambda.reshape(-1, 1)))
    weighted_thetas = np.multiply(thetas, pb.reshape(-1, 1))
    p = weighted_thetas[:, 0] / np.sum(weighted_thetas, axis=0)

    return {
        "measurement": measurement,
        "probabilities": p
    }


def prepare_and_measure_povm(shots, n):
    """
    Runs a prepare-and-measure classical simulation with a random POVM measurement

    Parameters
    ---------
    shots : int
        Number of shots the simulation is run with
    n: int
        Number of POVM random elements

    Returns
    -------
    dict
        A dictionary with the random state ('qubit'), random POVM measurement ('measurement') and
        the probabilities for each measurement outcome ('probabilities') in a nested structure including the
        theoretical probability ('born'), the execution runs ('runs') and the probability statistics ('stats')
    """

    # Alice prepares a random qubit
    qubit = qt.random.qubit()

    # Bob prepares a random measurement
    measurement = qt.random.povm(n)

    # import math
    # qubit = Qubit(np.array([(3 + 1.j * math.sqrt(3)) / 4., -0.5]))
    # # P4 = {1/2|0x0|, 1/2|1x1|, 1/2|+x+|, 1/2|-x-|}
    # proj = np.array([[[1, 0], [0, 0]], [[0, 0], [0, 1]], [[.5, .5], [.5, .5]], [[.5, -.5], [-.5, .5]]])
    # measurement = POVM(weights=0.5 * np.array([1, 1, 1, 1]), proj=proj)
    # n = measurement.size()

    experiment = {
        "qubit": qubit,
        "measurement": measurement,
        "probabilities": {
            "runs": np.zeros((shots, n)),
            "stats": np.zeros((n,)),
            "born": np.ones((n,))
        }
    }

    for i in range(shots):

        # Alice and Bob's shared randomness
        shared_randomness = np.array([qt.random.bloch_vector(), qt.random.bloch_vector()])

        # Alice prepares
        alice = prepare(shared_randomness, qubit)

        # Bob measures
        bob = measure_povm(shared_randomness, alice['bits'], measurement)

        # save simulation runs
        p = np.abs(bob['probabilities'])
        experiment['probabilities']['runs'][i, :] = p

        # accumulate counts according to Bob's probabilities
        index = random.choices(range(0, n), cum_weights=np.cumsum(p), k=1)[0]
        experiment['probabilities']['stats'][index] = experiment['probabilities']['stats'][index] + 1

    experiment['probabilities']['stats'] = experiment['probabilities']['stats'] / shots
    experiment['probabilities']['born'] = bob['measurement'].probability(qubit)

    return experiment
