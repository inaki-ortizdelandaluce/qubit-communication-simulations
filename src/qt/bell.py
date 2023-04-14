from enum import Enum
import numpy as np
import math
from qt.observable import Observable


class BellState(Enum):
    PHI_PLUS = 0
    PHI_MINUS = 1
    PSI_PLUS = 2
    PSI_MINUS = 3


class BellScenario:

    _states = {
        0: 1 / math.sqrt(2) * np.array([1, 0, 0, 1]),
        1: 1 / math.sqrt(2) * np.array([1, 0, 0, -1]),
        2: 1 / math.sqrt(2) * np.array([0, 1, 1, 0]),
        3: 1 / math.sqrt(2) * np.array([0, 1, -1, 0]),
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

        self.ket = self._states[state.value]
        self.alice = alice
        self.bob = bob
