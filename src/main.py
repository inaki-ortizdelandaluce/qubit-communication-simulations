import math
import matplotlib.pyplot as plt
from healpy.pixelfunc import ang2pix
import qt.classical
import qt.quantum
import qt.qubit
import qt.random
from qt.measurement import POVM
from qt.visualization import *



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

    counts = qt.quantum.prepare_and_measure_povm(1000000, qubit, povm)
    p = np.array([counts['00'], counts['01'], counts['10'], counts['11']])
    p = p / np.sum(p)
    print('Probabilities={}'.format(p))


if __name__ == "__main__":
    # test_random_states()
    # test_pvm_convergence()
    # test_random_povm()
    # test_povm_convergence()
    # test_povm_convergence_3d()
    # test_neumark()
    test_povm_circuit()
