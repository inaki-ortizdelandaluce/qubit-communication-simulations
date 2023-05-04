import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from healpy.pixelfunc import ang2pix
from scipy.special import rel_entr

import qt.classical
import qt.quantum
import qt.random

from qt.qubit import X, Y, Z, Qubit
from qt.bell import BellScenario, BellState
from qt.measurement import POVM
from qt.observable import Observable
from qt.visualization import *

from qiskit import transpile
from qiskit import execute, Aer, IBMQ
from qiskit.visualization import plot_histogram
from qiskit import QuantumCircuit
from qiskit.tools.monitor import job_monitor


def random_states():
    size = 100000
    n = 4
    pixels = 12 * n ** 2
    indexes = np.zeros(size)
    for i in range(size):
        theta, phi = qt.random.qubit().bloch_angles()
        pix = ang2pix(n, theta, phi)
        indexes[i] = pix

    # mpl.rcParams['font.family'] = 'Avenir'
    # plt.rcParams['font.size'] = 18
    # plt.rcParams['axes.linewidth'] = 2
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['font.family'] = 'STIXGeneral'

    fig, ax = plt.subplots()

    # ax.xaxis.set_tick_params(which='major', size=5, width=1, direction='in', top='on')
    # ax.xaxis.set_tick_params(which='minor', size=3, width=1, direction='in', top='on')
    # ax.yaxis.set_tick_params(which='major', size=5, width=1, direction='in', right='on')
    # ax.yaxis.set_tick_params(which='minor', size=3, width=1, direction='in', right='on')

    count, bins, ignored = ax.hist(indexes, bins=range(pixels + 1), density=True, fill=True, facecolor='whitesmoke',
                                   edgecolor='k', hatch='', linewidth=1, histtype='step')
    ax.plot(bins, np.ones_like(bins) / pixels, linewidth=2, color='b', linestyle='-', zorder=2)
    ax.set_xlabel('Pixel indices', labelpad=6)
    ax.set_xticks(np.append(np.arange(0, pixels, 25), pixels))
    ax.set_ylabel('Frequency', labelpad=6)
    # ax.set_yticklabels(ax.get_yticklabels() * pixels)
    ax.set_xlim(0, pixels)
    plt.show()
    return None


def random_povm():
    np.random.seed(0)
    q1 = qt.random.qubit()
    q2 = qt.random.qubit()

    povm = POVM.new(np.array([q1, q2]))
    elements = povm.elements

    for i in range(elements.shape[0]):
        print('\nE{} eigenvalues -> {}'.format(i, np.linalg.eig(elements[i])[0]))
        print('\nE{}=\n{}'.format(i, elements[i]))
        print('E{} >=0 > -> {}'.format(i, (np.all(np.linalg.eig(elements[i])[0] >= -np.finfo(np.float32).eps))))

    # print('Sum E_i = I -> {}'.format(np.allclose(np.identity(2), np.sum(e * a[:, np.newaxis, np.newaxis], axis=0))))
    # print('Sum E_i = I -> {}'.format(np.allclose(np.identity(2), np.tensordot(_e, _a, axes=([0], [0])))))
    print('Sum E_i = I -> {}'.format(np.allclose(np.identity(2), np.sum(elements, axis=0))))

    return None


def pm_pvm(shots):
    # run experiment
    np.random.seed(0)
    experiment = qt.classical.prepare_and_measure_pvm(shots)

    # plot probability convergence
    runs = experiment['probabilities']['runs']
    stats = experiment['probabilities']['stats']
    born = experiment['probabilities']['born']
    print('Stats:\np1={},p2={},pt={}'.format(stats[0], stats[1], np.sum(stats)))
    print('Born:\np1={},p2={},pt={}'.format(born[0], born[1], np.sum(born)))

    p = np.cumsum(runs[:, 0]) / (np.arange(len(runs[:, 0])) + 1)

    plt.plot(p)
    plt.axhline(y=born[0], color='r', linestyle='-')
    plt.show()
    return None


def pm_random(shots):
    # run experiment
    np.random.seed(1200)
    experiment = qt.classical.prepare_and_measure_povm(shots, 4)

    # plot probability convergence
    runs = experiment['probabilities']['runs']
    stats = experiment['probabilities']['stats']
    born = experiment['probabilities']['born']
    print('Stats:\np1={}, p2={}, p3={}, p4={}, pt={}'.format(stats[0], stats[1], stats[2], stats[3], np.sum(stats)))
    print('Born:\np1={}, p2={}, p3={}, p4={}, pt={}'.format(born[0], born[1], born[2], born[3], np.sum(born)))

    p1 = np.cumsum(runs[:, 0]) / (np.arange(len(runs[:, 0])) + 1)
    p2 = np.cumsum(runs[:, 1]) / (np.arange(len(runs[:, 1])) + 1)
    p3 = np.cumsum(runs[:, 2]) / (np.arange(len(runs[:, 2])) + 1)
    p4 = np.cumsum(runs[:, 3]) / (np.arange(len(runs[:, 3])) + 1)

    plt.plot(p1, color='r')
    plt.plot(p2, color='g')
    plt.plot(p3, color='b')
    plt.plot(p4, color='y')
    plt.axhline(y=born[0], color='r', linestyle='-')
    plt.axhline(y=born[1], color='g', linestyle='-')
    plt.axhline(y=born[2], color='b', linestyle='-')
    plt.axhline(y=born[3], color='y', linestyle='-')
    plt.title('Prepare and Measure with random state and POVM')
    plt.show()
    return None


def pm_trine(shots):
    # run experiment
    psi = qt.qubit.Qubit(np.array([(3 + 1.j * math.sqrt(3)) / 4., -0.5]))

    one = Qubit(np.array([1, 0])).rho()
    two = Qubit(0.5 * np.array([1, math.sqrt(3)])).rho()
    three = Qubit(0.5 * np.array([1, -math.sqrt(3)])).rho()
    povm = POVM(weights=2./3 * np.array([1, 1, 1]), proj=np.array([one, two, three], dtype=complex))

    experiment = qt.classical.prepare_and_measure_povm(shots, qubit=psi, measurement=povm)

    # plot probability convergence
    runs = experiment['probabilities']['runs']
    stats = experiment['probabilities']['stats']
    born = experiment['probabilities']['born']
    print('Stats:\np1={}, p2={}, p3={}, pt={}'.format(stats[0], stats[1], stats[2], np.sum(stats)))
    print('Born:\np1={}, p2={}, p3={}, pt={}'.format(born[0], born[1], born[2], np.sum(born)))

    p1 = np.cumsum(runs[:, 0]) / (np.arange(len(runs[:, 0])) + 1)
    p2 = np.cumsum(runs[:, 1]) / (np.arange(len(runs[:, 1])) + 1)
    p3 = np.cumsum(runs[:, 2]) / (np.arange(len(runs[:, 2])) + 1)

    plt.plot(p1, color='r')
    plt.plot(p2, color='g')
    plt.plot(p3, color='b')
    plt.axhline(y=born[0], color='r', linestyle='-')
    plt.axhline(y=born[1], color='g', linestyle='-')
    plt.axhline(y=born[2], color='b', linestyle='-')
    plt.title('Prepare and Measure with Trine POVM')
    plt.show()

    return None


def pm_cross(shots):
    # run experiment
    psi = qt.qubit.Qubit(np.array([(3 + 1.j * math.sqrt(3)) / 4., -0.5]))

    zero = np.array([[1, 0], [0, 0]])
    one = np.array([[0, 0], [0, 1]])
    plus = 0.5 * np.array([[1, 1], [1, 1]])
    minus = 0.5 * np.array([[1, -1], [-1, 1]])

    povm = POVM(weights=0.5 * np.array([1, 1, 1, 1]), proj=np.array([zero, one, plus, minus], dtype=complex))

    experiment = qt.classical.prepare_and_measure_povm(shots, qubit=psi, measurement=povm)

    # plot probability convergence
    runs = experiment['probabilities']['runs']
    stats = experiment['probabilities']['stats']
    born = experiment['probabilities']['born']
    print('Stats:\np1={}, p2={}, p3={}, pt={}'.format(stats[0], stats[1], stats[2], np.sum(stats)))
    print('Born:\np1={}, p2={}, p3={}, pt={}'.format(born[0], born[1], born[2], np.sum(born)))

    p1 = np.cumsum(runs[:, 0]) / (np.arange(len(runs[:, 0])) + 1)
    p2 = np.cumsum(runs[:, 1]) / (np.arange(len(runs[:, 1])) + 1)
    p3 = np.cumsum(runs[:, 2]) / (np.arange(len(runs[:, 2])) + 1)

    plt.plot(p1, color='r')
    plt.plot(p2, color='g')
    plt.plot(p3, color='b')
    plt.axhline(y=born[0], color='r', linestyle='-')
    plt.axhline(y=born[1], color='g', linestyle='-')
    plt.axhline(y=born[2], color='b', linestyle='-')
    plt.title('Prepare and Measure with Cross POVM')
    plt.show()

    return None


def pm_sic(shots):
    # run experiment
    psi = qt.qubit.Qubit(np.array([(3 + 1.j * math.sqrt(3)) / 4., -0.5]))

    one = Qubit(np.array([1, 0])).rho()
    two = Qubit(np.array([1/math.sqrt(3), math.sqrt(2/3)])).rho()
    three = Qubit(np.array([1/math.sqrt(3),
                            math.sqrt(2/3) * (math.cos(2 * math.pi/3) + 1.j * math.sin(2 * math.pi/3))])).rho()
    four = Qubit(np.array([1/math.sqrt(3),
                           math.sqrt(2/3) * (math.cos(4 * math.pi/3) + 1.j * math.sin(4 * math.pi/3))])).rho()

    povm = POVM(weights=0.5 * np.array([1, 1, 1, 1]), proj=np.array([one, two, three, four], dtype=complex))

    experiment = qt.classical.prepare_and_measure_povm(shots, qubit=psi, measurement=povm)

    # plot probability convergence
    runs = experiment['probabilities']['runs']
    stats = experiment['probabilities']['stats']
    born = experiment['probabilities']['born']
    print('Stats:\np1={}, p2={}, p3={}, p4={}, pt={}'.format(stats[0], stats[1], stats[2], stats[3], np.sum(stats)))
    print('Born:\np1={}, p2={}, p3={}, p4={}, pt={}'.format(born[0], born[1], born[2], born[3], np.sum(born)))

    p1 = np.cumsum(runs[:, 0]) / (np.arange(len(runs[:, 0])) + 1)
    p2 = np.cumsum(runs[:, 1]) / (np.arange(len(runs[:, 1])) + 1)
    p3 = np.cumsum(runs[:, 2]) / (np.arange(len(runs[:, 2])) + 1)
    p4 = np.cumsum(runs[:, 3]) / (np.arange(len(runs[:, 3])) + 1)

    plt.plot(p1, color='r')
    plt.plot(p2, color='g')
    plt.plot(p3, color='b')
    plt.plot(p4, color='y')
    plt.axhline(y=born[0], color='r', linestyle='-')
    plt.axhline(y=born[1], color='g', linestyle='-')
    plt.axhline(y=born[2], color='b', linestyle='-')
    plt.axhline(y=born[3], color='y', linestyle='-')
    plt.title('Prepare and Measure with SIC-POVM of 4 elements')
    plt.show()
    return None


def pm_random_3d():
    # run experiment
    shots = 10 ** 5
    experiment = qt.classical.prepare_and_measure_povm(shots, 4)

    # plot probability convergence
    runs = experiment['probabilities']['runs']
    stats = experiment['probabilities']['stats']
    born = experiment['probabilities']['born']
    print('Stats:\np1={}, p2={}, p3={}, p4={}, pt={}'.format(stats[0], stats[1], stats[2], stats[3], np.sum(stats)))
    print('Born:\np1={}, p2={}, p3={}, p4={}, pt={}'.format(born[0], born[1], born[2], born[3], np.sum(born)))

    p1 = np.cumsum(runs[:, 0]) / (np.arange(len(runs[:, 0])) + 1)
    p2 = np.cumsum(runs[:, 1]) / (np.arange(len(runs[:, 1])) + 1)
    p3 = np.cumsum(runs[:, 2]) / (np.arange(len(runs[:, 2])) + 1)
    p4 = np.cumsum(runs[:, 3]) / (np.arange(len(runs[:, 3])) + 1)

    fig = plt.figure(figsize=(10, 7))
    rect = [80000, 0.43, 20000, 0.20]  # left, bottom, width, height

    # main axes
    ax = fig.add_subplot(111)
    title = r'$|q\rangle = \frac{3 + i\sqrt{3}}{4}\;|0\rangle - \frac{1}{2}\;|1\rangle \quad P_{4}=\{\frac{1}{2}|0\rangle\langle0|,\;\frac{1}{2}|1\rangle\langle1|,\;\frac{1}{2}|+\rangle\langle+|,\;\frac{1}{2}|-\rangle\langle-|\}$'
    plt.title(title, fontsize='small')
    plt.xlabel('shots')
    plt.ylabel('probability')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    colors = ["#dc267f", "#648fff", "#fe6100", "#785ef0", "#ffb000"]

    ax.set_ylim([0, 1.0])
    ax.plot(p1, color=colors[0], linewidth=1)
    ax.plot(p2, color=colors[1], linewidth=1)
    ax.plot(p3, color=colors[2], linewidth=1)
    ax.plot(p4, color=colors[3], linewidth=1)
    ax.axhline(y=born[0], color=colors[0], linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=born[1], color=colors[1], linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=born[2], color=colors[2], linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=born[3], color=colors[3], linestyle='--', linewidth=1, alpha=0.5)

    # inset axes
    ax_inset = add_inset_axes(rect, units="ax", ax_target=ax, projection="3d")
    state = np.array([[-0.75, 0.43, 0.5]])
    measurements = np.array([[0, 0, 1.0], [0, 0, -1.0], [1.0, 0, 0], [-1.0, 0, 0]])
    plot_bloch_sphere(ax_inset, state, measurements)

    plt.show()
    return None


def neumark():
    psi = qt.qubit.Qubit(np.array([(3 + 1.j * math.sqrt(3)) / 4., -0.5]))
    print(psi.bloch_vector())

    zero = np.array([[1, 0], [0, 0]])
    one = np.array([[0, 0], [0, 1]])
    plus = 0.5 * np.array([[1, 1], [1, 1]])
    minus = 0.5 * np.array([[1, -1], [-1, 1]])

    povm = POVM(weights=0.5 * np.array([1, 1, 1, 1]), proj=np.array([zero, one, plus, minus], dtype=complex))
    unitary = povm.unitary()
    print(unitary)
    return None


def pm_circuit():
    qubit = qt.qubit.Qubit(np.array([(3 + 1.j * math.sqrt(3)) / 4., -0.5]))

    zero = np.array([[1, 0], [0, 0]])
    one = np.array([[0, 0], [0, 1]])
    plus = 0.5 * np.array([[1, 1], [1, 1]])
    minus = 0.5 * np.array([[1, -1], [-1, 1]])
    povm = POVM(weights=0.5 * np.array([1, 1, 1, 1]), proj=np.array([zero, one, plus, minus], dtype=complex))
    print(povm.unitary())

    shots = 10 ** 7
    results = qt.quantum.prepare_and_measure_povm(shots, qubit, povm)
    print('Probabilities={}'.format(results["probabilities"]))
    return None


def quantum_simulator():
    qc = QuantumCircuit(2, 2)

    psi = ((3 + 1.j * math.sqrt(3)) / 4., -0.5)

    U = [[0.70710678 + 0.j, 0. + 0.j, 0.70710678 + 0.j, 0. + 0.j],
         [0. + 0.j, 0.70710678 + 0.j, 0. + 0.j, 0.70710678 + 0.j],
         [0.5 - 0.j, 0.5 + 0.j, -0.5 + 0.j, -0.5 + 0.j],
         [-0.5 + 0.j, 0.5 + 0.j, 0.5 + 0.j, -0.5 + 0.j]]

    qc.initialize(psi, 0)
    qc.unitary(U, [0, 1])
    qc.measure([0, 1], [0, 1])
    qc.draw()

    backend = Aer.get_backend('aer_simulator')
    qc_transpiled = transpile(qc, backend)
    qc_transpiled.draw()

    job = backend.run(qc_transpiled, shots=4000)
    result = job.result()
    counts = result.get_counts(qc_transpiled)

    print(counts)
    plot_histogram(counts)

    sum(counts.values())
    print(counts['00'] / sum(counts.values()))
    print(counts['01'] / sum(counts.values()))
    print(counts['10'] / sum(counts.values()))
    print(counts['11'] / sum(counts.values()))


def quantum_computer():
    qc = QuantumCircuit(2, 2)

    psi = ((3 + 1.j * math.sqrt(3)) / 4., -0.5)

    U = [[0.70710678 + 0.j, 0. + 0.j, 0.70710678 + 0.j, 0. + 0.j],
         [0. + 0.j, 0.70710678 + 0.j, 0. + 0.j, 0.70710678 + 0.j],
         [- 0.5 + 0.j, -0.5 + 0.j, 0.5 + 0.j, 0.5 + 0.j],
         [-0.5 + 0.j, 0.5 + 0.j, 0.5 + 0.j, -0.5 + 0.j]]

    qc.initialize(psi, 0)
    qc.unitary(U, [0, 1])
    qc.measure([0, 1], [0, 1])
    qc.draw()

    IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
    qcomp = provider.get_backend('ibm_nairobi')
    # running in ibm_nairobi. 4000 shots

    qc_transpiled = transpile(qc, backend=qcomp)
    job = execute(qc_transpiled, backend=qcomp, shots=4000)
    job_monitor(job)
    result = job.result()
    counts = result.get_counts(qc_transpiled)
    plot_histogram(counts)
    sum(counts.values())
    print(counts['00'] / sum(counts.values()))
    print(counts['01'] / sum(counts.values()))
    print(counts['10'] / sum(counts.values()))
    print(counts['11'] / sum(counts.values()))


def kl_sample():
    import random
    from scipy.special import rel_entr
    import collections

    shots = 10 ** 7

    expected = np.array([0.375, 0.125, 0.0625, 0.4375])
    actual = np.array([0.275, 0.225, 0.0725, 0.4475])

    p = random.choices(np.arange(len(expected)), weights=expected, k=shots)
    q = random.choices(np.arange(len(expected)), weights=expected, k=shots)

    counter_p = collections.Counter(p)
    counter_q = collections.Counter(q)

    freq_p = np.array([counter_p[x] for x in sorted(counter_p.keys())])
    freq_p = freq_p / np.sum(freq_p)

    freq_q = np.array([counter_q[x] for x in sorted(counter_q.keys())])
    freq_q = freq_q / np.sum(freq_q)

    print(sum(rel_entr(expected, actual)))
    print(expected)
    print(freq_p)
    print(freq_q)
    return None


def pm_kl_classical_born():
    """
    Runs PM classical protocol and plots Kullback-Leibler divergence among classical and Born probability distributions
    """

    # run experiment
    np.random.seed(0)
    shots = 10 ** 4

    qubit = qt.qubit.Qubit(np.array([(3 + 1.j * math.sqrt(3)) / 4., -0.5]))
    # P4 = {1/2|0x0|, 1/2|1x1|, 1/2|+x+|, 1/2|-x-|}
    proj = np.array([[[1, 0], [0, 0]], [[0, 0], [0, 1]], [[.5, .5], [.5, .5]], [[.5, -.5], [-.5, .5]]])
    measurement = POVM(weights=0.5 * np.array([1, 1, 1, 1]), proj=proj)

    experiment = qt.classical.prepare_and_measure_povm(shots, 4, qubit=qubit, measurement=measurement)

    # plot Kullback-Leibler divergence
    runs = experiment['probabilities']['runs']
    stats = experiment['probabilities']['stats']
    born = experiment['probabilities']['born']
    print('Stats:\np1={}, p2={}, p3={}, p4={}, pt={}'.format(stats[0], stats[1], stats[2], stats[3], np.sum(stats)))
    print('Born:\np1={}, p2={}, p3={}, p4={}, pt={}'.format(born[0], born[1], born[2], born[3], np.sum(born)))

    p1 = np.cumsum(runs[:, 0]) / (np.arange(len(runs[:, 0])) + 1)
    p2 = np.cumsum(runs[:, 1]) / (np.arange(len(runs[:, 1])) + 1)
    p3 = np.cumsum(runs[:, 2]) / (np.arange(len(runs[:, 2])) + 1)
    p4 = np.cumsum(runs[:, 3]) / (np.arange(len(runs[:, 3])) + 1)

    actual = np.vstack((p1, p2, p3, p4))
    expected = np.repeat(born.reshape(born.shape[0], 1), actual.shape[1], axis=1)

    rows, cols = actual.shape
    kl = np.zeros((cols,))
    for i in range(cols):
        kl[i] = sum(rel_entr(expected[:, i], actual[:, i]))

    plt.plot(kl, color='b')
    plt.show()
    return None


def pm_kl_classical_quantum_simulator():
    """
    Runs classical protocol and quantum simulator and plots Kullback-Leibler divergence among classical and
    quantum simulator probability distribution
    """
    shots = 10 ** 4

    qubit = qt.qubit.Qubit(np.array([(3 + 1.j * math.sqrt(3)) / 4., -0.5]))
    # P4 = {1/2|0x0|, 1/2|1x1|, 1/2|+x+|, 1/2|-x-|}
    proj = np.array([[[1, 0], [0, 0]], [[0, 0], [0, 1]], [[.5, .5], [.5, .5]], [[.5, -.5], [-.5, .5]]], dtype=complex)
    measurement = POVM(weights=0.5 * np.array([1, 1, 1, 1]), proj=proj)

    # run classical protocol
    experiment1 = qt.classical.prepare_and_measure_povm(shots, qubit=qubit, measurement=measurement)
    runs = experiment1['probabilities']['runs']
    born = experiment1['probabilities']['born']
    p1 = np.cumsum(runs[:, 0]) / (np.arange(len(runs[:, 0])) + 1)
    p2 = np.cumsum(runs[:, 1]) / (np.arange(len(runs[:, 1])) + 1)
    p3 = np.cumsum(runs[:, 2]) / (np.arange(len(runs[:, 2])) + 1)
    p4 = np.cumsum(runs[:, 3]) / (np.arange(len(runs[:, 3])) + 1)
    experimental1 = np.vstack((p1, p2, p3, p4))
    theoretical = np.repeat(born.reshape(born.shape[0], 1), experimental1.shape[1], axis=1)
    print('Classical protocol executed')

    # run quantum circuit in Qiskit simulator
    experiment2 = qt.quantum.prepare_and_measure_povm(shots, qubit, measurement)
    import collections
    memory = experiment2["memory"]
    experimental2 = np.zeros(experimental1.shape)
    for i in range(len(memory)):
        summary = collections.Counter(memory[0: i + 1])
        summary = np.array([summary[k] for k in sorted(summary.keys())])
        p = np.zeros((measurement.size(),))
        p[:summary.shape[0]] = summary / np.sum(summary)
        experimental2[:, i] = p
    print('Quantum Circuit executed')

    # plot kl divergence
    _, cols = experimental1.shape
    klte1 = np.zeros((cols,))
    klte2 = np.zeros((cols,))
    klee = np.zeros((cols,))

    for i in range(cols):
        klte1[i] = sum(rel_entr(theoretical[:, i], experimental1[:, i]))
        klte2[i] = sum(rel_entr(theoretical[:, i], experimental2[:, i]))
        klee[i] = sum(rel_entr(experimental2[:, i], experimental1[:, i]))

    plt.title('Kullback-Leibler divergence {:.0E} shots'.format(shots))
    plt.plot(klte1, color='b', label='Born vs Classical Protocol')
    plt.plot(klte2, color='r', label='Born vs Quantum Simulator')
    # plt.plot(klee, color='g', label='Quantum Simulator vs Classical Protocol')
    plt.legend()
    plt.show()
    return None


def pm_kl_classical_quantum_simulator_born(shots):
    np.random.seed(1000)
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['font.family'] = 'STIXGeneral'

    fig, ax = plt.subplots(1, 1, layout='constrained')

    ax.xaxis.set_tick_params(which='major', size=5, width=1, direction='in', top='on')
    ax.xaxis.set_tick_params(which='minor', size=3, width=1, direction='in', top='on')
    ax.yaxis.set_tick_params(which='major', size=5, width=1, direction='in', right='on')
    ax.yaxis.set_tick_params(which='minor', size=3, width=1, direction='in', right='on')

    fig.suptitle(r'Cross-POVM')
    fig.supxlabel('Number of shots')
    fig.supylabel('Kullback-Leibler divergence')

    qubit = qt.qubit.Qubit(np.array([(3 + 1.j * math.sqrt(3)) / 4., -0.5]))
    proj = np.array([[[1, 0], [0, 0]], [[0, 0], [0, 1]], [[.5, .5], [.5, .5]], [[.5, -.5], [-.5, .5]]], dtype=complex)
    measurement = POVM(weights=0.5 * np.array([1, 1, 1, 1]), proj=proj)

    # run classical protocol
    experiment1 = qt.classical.prepare_and_measure_povm(shots, qubit=qubit, measurement=measurement)
    runs = experiment1['probabilities']['runs']
    stats = experiment1['probabilities']['stats']
    born = experiment1['probabilities']['born']
    p1 = np.cumsum(runs[:, 0]) / (np.arange(len(runs[:, 0])) + 1)
    p2 = np.cumsum(runs[:, 1]) / (np.arange(len(runs[:, 1])) + 1)
    p3 = np.cumsum(runs[:, 2]) / (np.arange(len(runs[:, 2])) + 1)
    p4 = np.cumsum(runs[:, 3]) / (np.arange(len(runs[:, 3])) + 1)
    experimental1 = np.vstack((p1, p2, p3, p4))
    theoretical = np.repeat(born.reshape(born.shape[0], 1), experimental1.shape[1], axis=1)
    print('Stats:p1={}, p2={}, p3={}, p4={}, pt={}'.format(stats[0], stats[1], stats[2], stats[3], np.sum(stats)))
    print('Born:p1={}, p2={}, p3={}, p4={}, pt={}'.format(born[0], born[1], born[2], born[3], np.sum(born)))
    print('Classical protocol executed')

    # run quantum circuit in Qiskit simulator
    experiment2 = qt.quantum.prepare_and_measure_povm(shots, qubit=qubit, povm=measurement)
    import collections
    memory = experiment2["memory"]
    experimental2 = np.zeros(experimental1.shape)
    for i in range(len(memory)):
        summary = collections.Counter(memory[0: i + 1])
        summary = np.array([summary[k] for k in sorted(summary.keys())])
        p = np.zeros((measurement.size(),))
        p[:summary.shape[0]] = summary / np.sum(summary)
        experimental2[:, i] = p
    print('Stats={}'.format(experiment2["probabilities"]))
    print('Quantum Circuit executed')

    # plot kl divergence
    _, cols = experimental1.shape
    klte1 = np.zeros((cols,))
    klte2 = np.zeros((cols,))
    kle1t = np.zeros((cols,))
    kle1e2 = np.zeros((cols,))

    for i in range(cols):
        klte1[i] = sum(rel_entr(theoretical[:, i], experimental1[:, i]))
        klte2[i] = sum(rel_entr(theoretical[:, i], experimental2[:, i]))
        kle1t[i] = sum(rel_entr(experimental1[:, i], theoretical[:, i]))
        kle1e2[i] = sum(rel_entr(experimental1[:, i], experimental2[:, i]))

    ax.plot(klte1, color='b', label='Born vs. Classical Protocol', linewidth=2)
    ax.plot(klte2, color='r', label='Born vs. Quantum Simulator', linewidth=2, linestyle='-', alpha=0.7)
    ax.plot(kle1e2, color='g', label='Classical Protocol vs. Quantum Simulator', linewidth=2, linestyle='-', alpha=0.7)
    ax.legend()

    plt.show()
    return None


def pm_kl_multiplot(shots):
    np.random.seed(0)
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['font.family'] = 'STIXGeneral'

    cases = ['Random-PVM', 'Cross-POVM', 'Trine-POVM', 'SIC-POVM', r"Random-POVM$^{1}$", r"Random-POVM$^{2}$"]

    fig, axs = plt.subplots(3, 2, figsize=(8, 10), layout='constrained')

    for ax, title in zip(axs.flat, cases):
        ax.xaxis.set_tick_params(which='major', size=5, width=1, direction='in', top='on')
        ax.xaxis.set_tick_params(which='minor', size=3, width=1, direction='in', top='on')
        ax.yaxis.set_tick_params(which='major', size=5, width=1, direction='in', right='on')
        ax.yaxis.set_tick_params(which='minor', size=3, width=1, direction='in', right='on')
        # ax.set_ylabel(r'$D_{KL}$', labelpad=2)
        ax.set_title(title)

    fig.supxlabel('Number of shots')
    fig.supylabel('Kullback-Leibler divergence')

    qubit = qt.qubit.Qubit(np.array([(3 + 1.j * math.sqrt(3)) / 4., -0.5]))

    # EXPERIMENT 1: RANDOM-PVM
    experiment1 = qt.classical.prepare_and_measure_pvm(shots)

    qubit1 = experiment1['qubit']
    runs1 = experiment1['probabilities']['runs']
    stats1 = experiment1['probabilities']['stats']
    born1 = experiment1['probabilities']['born']
    print('EXPERIMENT 1:\nState:{}\nPVM:{}'.format(qubit1, 'RANDOM-PVM'))
    print('Stats:p1={}, p2={}, pt={}'.format(stats1[0], stats1[1], np.sum(stats1)))
    print('Born:p1={}, p2={}, pt={}'.format(born1[0], born1[1], np.sum(born1)))

    p11 = np.cumsum(runs1[:, 0]) / (np.arange(len(runs1[:, 0])) + 1)
    p12 = np.cumsum(runs1[:, 1]) / (np.arange(len(runs1[:, 1])) + 1)

    actual1 = np.vstack((p11, p12))
    expected1 = np.repeat(born1.reshape(born1.shape[0], 1), actual1.shape[1], axis=1)
    plot_kl(axs[0][0], actual1, expected1)

    # EXPERIMENT 2: CROSS-POVM
    # P4 = {1/2|0x0|, 1/2|1x1|, 1/2|+x+|, 1/2|-x-|}
    proj2 = np.array([[[1, 0], [0, 0]], [[0, 0], [0, 1]], [[.5, .5], [.5, .5]], [[.5, -.5], [-.5, .5]]])
    measurement2 = POVM(weights=0.5 * np.array([1, 1, 1, 1]), proj=proj2)
    experiment2 = qt.classical.prepare_and_measure_povm(shots, 4, qubit=qubit, measurement=measurement2)

    runs2 = experiment2['probabilities']['runs']
    stats2 = experiment2['probabilities']['stats']
    born2 = experiment2['probabilities']['born']
    print('EXPERIMENT 2:\nState:{}\nPOVM:{}'.format(qubit, 'CROSS-POVM'))
    print('Stats:p1={}, p2={}, p3={}, p4={}, pt={}'.format(stats2[0], stats2[1], stats2[2], stats2[3], np.sum(stats2)))
    print('Born:p1={}, p2={}, p3={}, p4={}, pt={}'.format(born2[0], born2[1], born2[2], born2[3], np.sum(born2)))

    p21 = np.cumsum(runs2[:, 0]) / (np.arange(len(runs2[:, 0])) + 1)
    p22 = np.cumsum(runs2[:, 1]) / (np.arange(len(runs2[:, 1])) + 1)
    p23 = np.cumsum(runs2[:, 2]) / (np.arange(len(runs2[:, 2])) + 1)
    p24 = np.cumsum(runs2[:, 3]) / (np.arange(len(runs2[:, 3])) + 1)

    actual2 = np.vstack((p21, p22, p23, p24))
    expected2 = np.repeat(born2.reshape(born2.shape[0], 1), actual2.shape[1], axis=1)
    plot_kl(axs[0][1], actual2, expected2)

    # EXPERIMENT 3: TRINE-POVM
    one = Qubit(np.array([1, 0])).rho()
    two = Qubit(0.5 * np.array([1, math.sqrt(3)])).rho()
    three = Qubit(0.5 * np.array([1, -math.sqrt(3)])).rho()
    measurement3 = POVM(weights=2. / 3 * np.array([1, 1, 1]), proj=np.array([one, two, three], dtype=complex))
    experiment3 = qt.classical.prepare_and_measure_povm(shots, qubit=qubit, measurement=measurement3)

    runs3 = experiment3['probabilities']['runs']
    stats3 = experiment3['probabilities']['stats']
    born3 = experiment3['probabilities']['born']
    print('EXPERIMENT 3:\nState:{}\nPOVM:{}'.format(qubit, 'TRINE-POVM'))
    print('Stats:p1={}, p2={}, p3={}, pt={}'.format(stats3[0], stats3[1], stats3[2], np.sum(stats3)))
    print('Born:p1={}, p2={}, p3={}, pt={}'.format(born3[0], born3[1], born3[2], np.sum(born3)))

    p31 = np.cumsum(runs3[:, 0]) / (np.arange(len(runs3[:, 0])) + 1)
    p32 = np.cumsum(runs3[:, 1]) / (np.arange(len(runs3[:, 1])) + 1)
    p33 = np.cumsum(runs3[:, 2]) / (np.arange(len(runs3[:, 2])) + 1)

    actual3 = np.vstack((p31, p32, p33))
    expected3 = np.repeat(born3.reshape(born3.shape[0], 1), actual3.shape[1], axis=1)
    plot_kl(axs[1][0], actual3, expected3)

    # EXPERIMENT 4: SIC-POVM
    one = Qubit(np.array([1, 0])).rho()
    two = Qubit(np.array([1/math.sqrt(3), math.sqrt(2/3)])).rho()
    three = Qubit(np.array([1/math.sqrt(3),
                            math.sqrt(2/3) * (math.cos(2 * math.pi/3) + 1.j * math.sin(2 * math.pi/3))])).rho()
    four = Qubit(np.array([1/math.sqrt(3),
                           math.sqrt(2/3) * (math.cos(4 * math.pi/3) + 1.j * math.sin(4 * math.pi/3))])).rho()
    measurement4 = POVM(weights=0.5 * np.array([1, 1, 1, 1]), proj=np.array([one, two, three, four], dtype=complex))
    experiment4 = qt.classical.prepare_and_measure_povm(shots, qubit=qubit, measurement=measurement4)

    runs4 = experiment4['probabilities']['runs']
    stats4 = experiment4['probabilities']['stats']
    born4 = experiment4['probabilities']['born']
    print('EXPERIMENT 4:\nState:{}\nPOVM:{}'.format(qubit, 'SIC-POVM'))
    print('Stats:p1={}, p2={}, p3={}, p4={}, pt={}'.format(stats4[0], stats4[1], stats4[2], stats4[3], np.sum(stats4)))
    print('Born:p1={}, p2={}, p3={}, p4={}, pt={}'.format(born4[0], born4[1], born4[2], born4[2], np.sum(born3)))

    p41 = np.cumsum(runs4[:, 0]) / (np.arange(len(runs4[:, 0])) + 1)
    p42 = np.cumsum(runs4[:, 1]) / (np.arange(len(runs4[:, 1])) + 1)
    p43 = np.cumsum(runs4[:, 2]) / (np.arange(len(runs4[:, 2])) + 1)
    p44 = np.cumsum(runs4[:, 3]) / (np.arange(len(runs4[:, 3])) + 1)

    actual4 = np.vstack((p41, p42, p43, p44))
    expected4 = np.repeat(born4.reshape(born4.shape[0], 1), actual4.shape[1], axis=1)
    plot_kl(axs[1][1], actual4, expected4)

    # EXPERIMENT 5: RANDOM-POVM-1
    experiment5 = qt.classical.prepare_and_measure_povm(shots, 4)

    qubit5 = experiment5['qubit']
    runs5 = experiment5['probabilities']['runs']
    stats5 = experiment5['probabilities']['stats']
    born5 = experiment5['probabilities']['born']
    print('EXPERIMENT 5:\nState:{}\nPOVM:{}'.format(qubit5, 'RANDOM-POVM-1'))
    print('Stats:p1={}, p2={}, p3={}, p4={}, pt={}'.format(stats5[0], stats5[1], stats5[2], stats5[3], np.sum(stats5)))
    print('Born:p1={}, p2={}, p3={}, p4={}, pt={}'.format(born5[0], born5[1], born5[2], born5[2], np.sum(born5)))

    p51 = np.cumsum(runs5[:, 0]) / (np.arange(len(runs5[:, 0])) + 1)
    p52 = np.cumsum(runs5[:, 1]) / (np.arange(len(runs5[:, 1])) + 1)
    p53 = np.cumsum(runs5[:, 2]) / (np.arange(len(runs5[:, 2])) + 1)
    p54 = np.cumsum(runs5[:, 3]) / (np.arange(len(runs5[:, 3])) + 1)

    actual5 = np.vstack((p51, p52, p53, p54))
    expected5 = np.repeat(born5.reshape(born5.shape[0], 1), actual5.shape[1], axis=1)
    plot_kl(axs[2][0], actual5, expected5)

    # EXPERIMENT 6: RANDOM-POVM-2
    experiment6 = qt.classical.prepare_and_measure_povm(shots, 4)

    qubit6 = experiment6['qubit']
    runs6 = experiment6['probabilities']['runs']
    stats6 = experiment6['probabilities']['stats']
    born6 = experiment6['probabilities']['born']
    print('EXPERIMENT 6:\nState:{}\nPOVM:{}'.format(qubit6, 'RANDOM-POVM-2'))
    print('Stats:p1={}, p2={}, p3={}, p4={}, pt={}'.format(stats6[0], stats6[1], stats6[2], stats6[3], np.sum(stats6)))
    print('Born:p1={}, p2={}, p3={}, p4={}, pt={}'.format(born6[0], born6[1], born6[2], born6[2], np.sum(born6)))

    p61 = np.cumsum(runs6[:, 0]) / (np.arange(len(runs6[:, 0])) + 1)
    p62 = np.cumsum(runs6[:, 1]) / (np.arange(len(runs6[:, 1])) + 1)
    p63 = np.cumsum(runs6[:, 2]) / (np.arange(len(runs6[:, 2])) + 1)
    p64 = np.cumsum(runs6[:, 3]) / (np.arange(len(runs6[:, 3])) + 1)

    actual6 = np.vstack((p61, p62, p63, p64))
    expected6 = np.repeat(born6.reshape(born6.shape[0], 1), actual6.shape[1], axis=1)
    plot_kl(axs[2][1], actual6, expected6)

    plt.show()
    return None


def plot_kl(ax, actual, expected, label='', color='b', linewidth=1, linestyle='-'):
    rows, cols = actual.shape
    kl = np.zeros((cols,))
    for i in range(cols):
        kl[i] = sum(rel_entr(expected[:, i], actual[:, i]))
    # ax.plot(kl, color=color, label=label, linewidth=linewidth, linestyle=linestyle)
    # ax.legend()
    ax.plot(kl, color=color, linewidth=linewidth, linestyle=linestyle)


def chsh_sample():
    a0 = Observable(Z)
    a1 = Observable(X)
    b0 = Observable(-1 / math.sqrt(2) * (X + Z))
    b1 = Observable(1 / math.sqrt(2) * (X - Z))

    bell = BellScenario(BellState.PSI_MINUS, alice=(a0, a1), bob=(b0, b1))
    print('Expected CHSH={}'.format(bell.chsh()))

    actual = np.array([[0.4267916, 0.0731243, 0.073249, 0.4268351],
                       [0.4269109, 0.0731628, 0.0732563, 0.42667],
                       [0.426685, 0.0731767, 0.0732062, 0.4269321],
                       [0.0733566, 0.4265009, 0.4270081, 0.0731344]]).T
    sign = np.array([1, -1, -1, 1])
    e = np.sum(actual * sign[:, np.newaxis], axis=0)
    print('Actual CHSH={}'.format(abs(np.sum(e * np.array([1, 1, 1, -1])))))
    return None


def bell_sample_probabilities():
    np.random.seed(0)

    shots = 10 ** 7
    a0 = Observable(Z)
    a1 = Observable(X)
    b0 = Observable(-1 / math.sqrt(2) * (X + Z))
    b1 = Observable(1 / math.sqrt(2) * (X - Z))

    alice = Qubit(a0.eigenvector(1))
    bob = (Qubit(b0.eigenvector(1)), Qubit(b0.eigenvector(-1)))

    # experiment = qt.classical.bell_singlet(shots, alice, bob)
    # stats = experiment['probabilities']['stats']
    # print('Stats:\np1={},p2={},p3={},p4={}'.format(stats[0], stats[1], stats[2], stats[3]))

    experiment = qt.classical.bell_singlet_full(shots, alice=(a0, a1), bob=(b0, b1))
    stats = experiment['probabilities']['stats']
    born = experiment['probabilities']['born']
    print('Stats:\n{}'.format(stats))
    print('Born:\n{}'.format(born))
    return None


def bell_sample_heatmap():
    x = np.arange(0, 5)
    y = np.arange(0, 5)

    actual = np.array([[0.4267916, 0.0731243, 0.073249, 0.4268351],
                       [0.4269109, 0.0731628, 0.0732563, 0.42667],
                       [0.426685, 0.0731767, 0.0732062, 0.4269321],
                       [0.0733566, 0.4265009, 0.4270081, 0.0731344]]).T

    expected = np.array([[0.4267767, 0.0732233, 0.0732233, 0.4267767],
                         [0.4267767, 0.0732233, 0.0732233, 0.4267767],
                         [0.4267767, 0.0732233, 0.0732233, 0.4267767],
                         [0.0732233, 0.4267767, 0.4267767, 0.0732233]]).T

    outcomes = ["$(+1,+1)$", "$(+1,-1)$", "$(-1,+1)$", "$(-1,-1)$"]
    measurements = ["($A_0$,$B_0$)", "$(A_0,B_1)$", "$(A_1,B_0)$", "$(A_1,B_1)$"]

    fig, ax = plt.subplots(1, 2)
    # expected
    im = ax[0].imshow(expected, cmap='PiYG')

    ax[0].set_xticks(np.arange(len(measurements)), labels=measurements)
    ax[0].set_yticks(np.arange(len(outcomes)), labels=outcomes)

    for i in range(len(measurements)):
        for j in range(len(outcomes)):
            text = ax[0].text(j, i, expected[i, j],
                              ha="center", va="center", color="w")

    ax[0].set_title("Born\'s Rule ")

    # actual
    im = ax[1].imshow(actual, cmap='PiYG')

    ax[1].set_xticks(np.arange(len(measurements)), labels=measurements)
    ax[1].set_yticks(np.arange(len(outcomes)), labels=outcomes)

    for i in range(len(measurements)):
        for j in range(len(outcomes)):
            text = ax[1].text(j, i, actual[i, j],
                              ha="center", va="center", color="w")

    ax[1].set_title("Classical Protocol")

    # fig.suptitle('Bell scenario')

    fig.tight_layout()
    plt.show()
    return None


def bell():
    np.random.seed(0)

    shots = 10**5
    a0 = Observable(Z)
    a1 = Observable(X)
    b0 = Observable(-1 / math.sqrt(2) * (X + Z))
    b1 = Observable(1 / math.sqrt(2) * (X - Z))

    alice = Qubit(a0.eigenvector(1))
    bob = (Qubit(b0.eigenvector(1)), Qubit(b0.eigenvector(-1)))

    experiment = qt.classical.bell_singlet_full(shots, alice=(a0, a1), bob=(b0, b1))
    actual = experiment['probabilities']['stats'].T
    expected = np.array([[0.4267767, 0.0732233, 0.0732233, 0.4267767],
                         [0.4267767, 0.0732233, 0.0732233, 0.4267767],
                         [0.4267767, 0.0732233, 0.0732233, 0.4267767],
                         [0.0732233, 0.4267767, 0.4267767, 0.0732233]]).T

    x = np.arange(0, 5)
    y = np.arange(0, 5)

    outcomes = ["$(+1,+1)$", "$(+1,-1)$", "$(-1,+1)$", "$(-1,-1)$"]
    measurements = ["($A_0$,$B_0$)", "$(A_0,B_1)$", "$(A_1,B_0)$", "$(A_1,B_1)$"]

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    # expected
    im = ax[0].imshow(expected, cmap='PiYG')

    ax[0].set_xticks(np.arange(len(measurements)), labels=measurements)
    ax[0].set_yticks(np.arange(len(outcomes)), labels=outcomes)

    for i in range(len(measurements)):
        for j in range(len(outcomes)):
            text = ax[0].text(j, i, expected[i, j],
                              ha="center", va="center", color="w")

    ax[0].set_title("Born\'s Rule ")

    # actual
    im = ax[1].imshow(actual, cmap='PiYG')

    ax[1].set_xticks(np.arange(len(measurements)), labels=measurements)
    ax[1].set_yticks(np.arange(len(outcomes)), labels=outcomes)

    for i in range(len(measurements)):
        for j in range(len(outcomes)):
            text = ax[1].text(j, i, actual[i, j],
                              ha="center", va="center", color="w")

    ax[1].set_title("Classical Protocol - $10^{5}$ shots")

    fig.tight_layout()
    plt.show()
    return None


if __name__ == "__main__":
    shots = 10 ** 7
    # random_states()
    # random_povm()
    pm_pvm(shots)
    pm_random(shots)
    # pm_random_3d()
    pm_trine(shots)
    pm_cross(shots)
    pm_sic(shots)
    # neumark()
    # pm_circuit()
    # quantum_simulator()
    # quantum_computer()
    # kl_sample()
    # pm_kl_classical_born()
    # pm_kl_classical_quantum_simulator()
    pm_kl_classical_quantum_simulator_born(shots)
    # pm_kl_multiplot(10**4)
    # chsh_sample()
    # bell_sample_probabilities()
    # bell_sample_heatmap()
    # bell()



