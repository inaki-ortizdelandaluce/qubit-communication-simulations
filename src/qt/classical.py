import math
import numpy as np
import random
import qt.random
from qt.qubit import Qubit
from qt.measurement import PVM, POVM
from qt.bell import BellState, BellScenario
from qt.observable import Observable


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


def prepare_and_measure_povm(shots, n=4, qubit=None, measurement=None):
    """
    Runs a prepare-and-measure classical simulation with a random POVM measurement

    Parameters
    ---------
    shots : int
        Number of shots the simulation is run with

    n: int, optional
        Number of POVM random elements. Used if no measurement argument is specified, default value is 4.

    qubit : Qubit, optional
        Alice's qubit state. If not specified, a random qubit state will be used instead

    measurement: POVM, optional
        Bob's POVM measurement. If not specified a random POVM will be used instead


    Returns
    -------
    dict
        A dictionary with the random state ('qubit'), random POVM measurement ('measurement') and
        the probabilities for each measurement outcome ('probabilities') in a nested structure including the
        theoretical probability ('born'), the execution runs ('runs') and the probability statistics ('stats')
    """

    if qubit is None:
        # Alice prepares a random qubit
        qubit = qt.random.qubit()

    if measurement is None:
        # Bob prepares a random measurement
        measurement = qt.random.povm(n)

    n = measurement.size()

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


def bell_singlet_full(shots, alice, bob):
    """
    Runs a classical simulation on a Bell singlet state for a set of local observables

    Parameters
    ---------
    shots : int
        Number of shots the simulation is run with

    alice: tuple[Observable, Observable]
        Alice's local projective measurements described as a tuple of observables

    bob: tuple[Observable, Observable]
        Bob's local projective measurements described as a tuple of observables

    Returns
    -------
    dict
        A dictionary with the Bell state ('state'), Alice's and Bob's local projective measurements ('alice', 'bob')
        and the joint probabilities for each measurement outcome ('probabilities') in a nested structure including the
        theoretical probabilities ('born'), the execution runs ('runs') and the probability statistics ('stats')
    """

    if type(alice) is not tuple:
        raise ValueError('Alice\'s observables is not a valid tuple')
    elif len(alice) != 2:
        raise ValueError('Alice\'s number of observables is not valid:{}'.format(str(len(alice))))

    if type(bob) is not tuple:
        raise ValueError('Bob\'s observables is not a valid tuple')
    elif len(bob) != 2:
        raise ValueError('Bob\'s number of observables is not tuple:{}'.format(str(len(bob))))

    state = BellState.PSI_MINUS

    experiment = {
        "state": state,
        "alice": alice,
        "bob": bob,
        "probabilities": {
            "runs": np.zeros((shots, 4, 4)),
            "stats": np.zeros((4, 4)),
            "born": np.zeros((4, 4))
        }
    }

    # Compute theoretical probabilities
    bell = BellScenario(state, alice, bob)
    experiment['probabilities']['born'] = bell.probability().T

    for i in range(2):

        # Alice's state corresponding to the positive local projector
        a = Qubit(alice[i].eigenvector(1))

        for j in range(2):

            # Bob's states corresponding to the positive and negative local projectors
            b = (Qubit(bob[j].eigenvector(1)), Qubit(bob[j].eigenvector(-1)))

            ab = bell_singlet(shots, a, b)

            ij = int('{}{}'.format(i, j), 2)
            experiment['probabilities']['runs'][:, :, ij] = ab['probabilities']['runs']
            experiment['probabilities']['stats'][ij] = ab['probabilities']['stats']

    return experiment


def bell_singlet(shots, a, b):
    """
    Runs a classical simulation on a Bell singlet state for a set of states corresponding to local projection valued
    measurements

    Parameters
    ---------
    shots : int
        Number of shots the simulation is run with

    a: Qubit
        Alice's state corresponding to the positive local projection valued measurement operator

    b: tuple(Qubit, Qubit)
        Bob's states corresponding to the positive and negative local projection valued measurement operators

    Returns
    -------
    dict
        A dictionary with the joint probabilities for each measurement outcome ('probabilities') in a nested structure
        including the execution runs ('runs') and the probability statistics ('stats')
    """

    experiment = {
        "probabilities": {
            "runs": np.zeros((shots, 4)),
            "stats": np.zeros((4,))
        }
    }

    # Alice's positive local projector as bloch vector
    x = np.asarray([a.bloch_vector()])

    # Bob's local projectors as bloch vectors
    y = np.asarray([b[0].bloch_vector(), b[1].bloch_vector()])

    for i in range(shots):

        # Alice and Bob's shared randomness
        lambda1, lambda2 = qt.random.bloch_vector(), qt.random.bloch_vector()

        # Alice performs local projective measurements
        a = - np.sign(x @ lambda1)

        # Alice sends bit to Bob
        c = -a * np.sign(x @ lambda2)

        # Bob flips the lambda if c = -1
        lambda2 = c * lambda2

        # compute lambda according to the probabilities {pb}
        lambdas = np.array([lambda1, lambda2])

        pb = 0.5 * np.array([1, 1])
        index = random.choices(range(0, 2), cum_weights=np.cumsum(pb), k=1)[0]
        ly = np.abs(np.matmul(lambdas, y.T))
        _lambda = lambdas[np.argmax(ly, axis=0)[index]]

        # compute probabilities
        thetas = theta(np.matmul(y, _lambda.reshape(-1, 1)))
        weighted_thetas = np.multiply(thetas, pb.reshape(-1, 1))
        p = weighted_thetas[:, 0] / np.sum(weighted_thetas, axis=0)

        bits = '{}{}'.format(int(0.5 * (1 - a[0])), np.where(p == 1)[0][0])
        index = int(bits, 2)
        experiment['probabilities']['stats'][index] += 1

    experiment['probabilities']['stats'] = experiment['probabilities']['stats'] / shots

    return experiment
