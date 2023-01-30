import numpy as np
import cmath
import math


class Qubit:
    def __init__(self, a=1, b=0):
        """
        Initializes a qubit in the computational basis. If no arguments are provided, it returns the zero state.

        Parameters
        ---------
        a : complex
            The amplitude of the zero state
        b : complex
            The amplitude of the one state


        """
        self.zero = complex(a)
        self.one = complex(b)
        self.normalize()

    def __repr__(self):
        return '{}|0> + {}|1>'.format(self.zero, self.one)

    @classmethod
    def from_array(cls, arr):
        """
        Creates a Qubit instance from a two-dimensional array.

        Parameters
        ---------
        arr : ndarray
            Two-dimensional complex array.

        Returns
        -------
        Qubit
            The Qubit instance.
        """
        return cls(arr[0], arr[1])

    def to_array(self):
        return np.array([self.zero, self.one], dtype=np.complex_)

    def normalize(self):
        arr = self.to_array()
        self.zero, self.one = arr/np.linalg.norm(arr)

    def bloch_angles(self):
        """
        Return the Bloch sphere coordinates in a tuple with polar and azimuthal angles in radians.

        Returns
        -------
        (float, float)
            The Bloch sphere coordinates, first the polar angle and then the azimuthal angle (both in radians).
        """
        r0, phi0 = cmath.polar(self.zero)
        r1, phi1 = cmath.polar(self.one)
        theta = 2 * math.acos(r0)
        phi = phi1 - phi0

        return theta, phi


class TwoQubit:
    def __init__(self, q1, q2):
        """
            Initializes two qubits
        """
        self.q1 = q1
        self.q2 = q2

