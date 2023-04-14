import math
import matplotlib.pyplot as plt
from healpy.pixelfunc import ang2pix
import qt.classical
import qt.quantum
import qt.qubit
import qt.random
from qt.measurement import POVM
from qt.visualization import *
from scipy.special import rel_entr


def test_random_states():
    size = 100000
    n = 4
    pixels = 12 * n ** 2
    indexes = np.zeros(size)
    for i in range(size):
        theta, phi = qt.random.qubit().bloch_angles()
        pix = ang2pix(n, theta, phi)
        indexes[i] = pix

    count, bins, ignored = plt.hist(indexes, bins=range(pixels + 1), density=True)
    plt.plot(bins, np.ones_like(bins) / pixels, linewidth=2, color='r')
    plt.show()
    return None


def test_random_povm():
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


def test_pvm_convergence():
    # run experiment
    np.random.seed(0)
    shots = 10 ** 7
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


def test_povm_convergence():
    # run experiment
    np.random.seed(0)
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

    plt.plot(p1, color='r')
    plt.plot(p2, color='g')
    plt.plot(p3, color='b')
    plt.plot(p4, color='y')
    plt.axhline(y=born[0], color='r', linestyle='-')
    plt.axhline(y=born[1], color='g', linestyle='-')
    plt.axhline(y=born[2], color='b', linestyle='-')
    plt.axhline(y=born[3], color='y', linestyle='-')
    plt.show()
    return None


def test_povm_convergence_3d():
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


def test_neumark():
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


def test_povm_circuit():
    qubit = qt.qubit.Qubit(np.array([(3 + 1.j * math.sqrt(3)) / 4., -0.5]))

    zero = np.array([[1, 0], [0, 0]])
    one = np.array([[0, 0], [0, 1]])
    plus = 0.5 * np.array([[1, 1], [1, 1]])
    minus = 0.5 * np.array([[1, -1], [-1, 1]])
    povm = POVM(weights=0.5 * np.array([1, 1, 1, 1]), proj=np.array([zero, one, plus, minus], dtype=complex))
    print(povm.unitary())

    shots = 10**7
    results = qt.quantum.prepare_and_measure_povm(shots, qubit, povm)
    print('Probabilities={}'.format(results["probabilities"]))
    return None


def test_probability_sampling():
    import random
    from scipy.special import rel_entr
    import collections

    shots = 10**7

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


def test_kl_classical_born():
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
    kl = np.zeros((cols, ))
    for i in range(cols):
        kl[i] = sum(rel_entr(expected[:, i], actual[:, i]))

    plt.plot(kl, color='b')
    plt.show()
    return None


def test_kl_classical_quantum_simulator():
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
        summary = collections.Counter(memory[0: i+1])
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


def test_bell():
    A0 = np.array([[0, 1],[1, 0]])
    A1 = np.array([[1, 0], [0, -1]])
    B0 = 1 / math.sqrt(2) * np.array([[-1, -1], [-1, 1]])
    B1 = 1 / math.sqrt(2) * np.array([[-1, 1], [1, 1]])

    w, v = np.linalg.eig(A0)
    q0 = qt.qubit.Qubit(v[:, 0])
    q1 = qt.qubit.Qubit(v[:, 1])


if __name__ == "__main__":
    # test_random_states()
    # test_pvm_convergence()
    # test_random_povm()
    # test_povm_convergence()
    # test_povm_convergence_3d()
    # test_neumark()
    # test_povm_circuit()
    # test_probability_sampling()
    # test_kl_classical_born()
    test_kl_classical_quantum_simulator()
