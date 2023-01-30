import pytest
import numpy as np
import math


@pytest.fixture
def qubit_array():
    return 1/math.sqrt(2) * np.array([[1, 1]])
