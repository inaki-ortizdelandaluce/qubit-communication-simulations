from enum import Enum
import numpy as np
import math
from qt.observable import Observable
from qt.qudit import Qudit
from qt.qudit import Qubit


class BellState(Enum):
    PHI_PLUS = 0
    PHI_MINUS = 1
    PSI_PLUS = 2
    PSI_MINUS = 3


class BellScenario:

    _states = {
        0: Qudit(1 / math.sqrt(2) * np.array([1, 0, 0, 1])),
        1: Qudit(1 / math.sqrt(2) * np.array([1, 0, 0, -1])),
        2: Qudit(1 / math.sqrt(2) * np.array([0, 1, 1, 0])),
        3: Qudit(1 / math.sqrt(2) * np.array([0, 1, -1, 0]))
    }

    def __init__(self, state: BellState, alice: tuple[Observable, Observable], bob: tuple[Observable, Observable]):

        if type(alice) is not tuple:
            raise ValueError('Alice\'s observables is not a valid tuple')
        elif len(alice) != 2:
            raise ValueError('Alice\'s number of observables is not valid:{}'.format(str(len(alice))))

        if type(bob) is not tuple:
            raise ValueError('Bob\'s observables is not a valid tuple')
        elif len(bob) != 2:
            raise ValueError('Bob\'s number of observables is not tuple:{}'.format(str(len(bob))))

        self.state = self._states[state.value]
        self.alice = alice
        self.bob = bob

    def probability(self):
        p = np.zeros((4, 4), dtype=float)
        for u in range(4):
            for v in range(4):
                i, j, m, n = (-1) ** (u >> 1), (-1) ** (u % 2), v >> 1, v % 2

                a = Qubit(self.alice[m].eigenvector(i))
                b = Qubit(self.bob[n].eigenvector(j))
                ab = Qudit.bipartite(a, b)
                p[u, v] = np.real(np.trace(np.matmul(ab.rho(), self.state.rho())))
                # print('p{}{}=({},{})x(A{},B{})={}'.format(u, v, i, j, m, n, p[u, v]))
        return p

    def expected_values(self):
        p = self.probability()
        sign = np.array([1, -1, -1, 1])
        return np.sum(p * sign[:, np.newaxis], axis=0)

    def chsh(self):
        e = self.expected_values()
        return abs(np.sum(e * np.array([1, 1, 1, -1])))
