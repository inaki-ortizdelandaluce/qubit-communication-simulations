import numpy as np
import cmath
import math


class Qubit:
    def __init__(self, alpha=1, beta=0):
        """
        Initializes a qubit in the computational basis. If no arguments are provided, it returns the zero state.

        Parameters
        ---------
        alpha : complex
            The amplitude of the zero state.
        beta : complex
            The amplitude of the one state.
        """
        self.alpha = complex(alpha)
        self.beta = complex(beta)
        self.normalize()

    def __repr__(self):
        return '{} |0> + {} |1>'.format(self.alpha, self.beta)

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
        return np.array([self.alpha, self.beta], dtype=np.complex_)

    def normalize(self):
        arr = self.to_array()
        self.alpha, self.beta = arr/np.linalg.norm(arr)

    def bloch_angles(self):
        """
        Return the spherical coordinates of the qubit in the Bloch sphere, with polar and azimuthal angles in radians.

        Returns
        -------
        (float, float)
            The Bloch sphere coordinates, first the polar angle and then the azimuthal angle (both in radians).
        """
        r0, phi0 = cmath.polar(self.alpha)
        r1, phi1 = cmath.polar(self.beta)
        theta = 2 * math.acos(r0)
        phi = phi1 - phi0

        return theta, phi

    def bloch_vector(self):
        """
         Return the cartesian coordinates of the qubit in the Bloch sphere.

         Returns
         -------
         (float, float, float)
             The cartesian coordinates of the qubit in the Bloch sphere (xyz).

         """
        theta, phi = self.bloch_angles()

        x = math.sin(theta) * math.cos(phi)
        y = math.sin(theta) * math.sin(phi)
        z = math.cos(theta)

        return x, y, z

    def to_density_matrix(self):
        """
         Return the density matrix corresponding to the qubit in a pure state.

         Returns
         -------
         ndarray
             A 2x2 density matrix corresponding to the qubit in a pure state.
         """
        return np.outer(self.to_array(), self.to_array().conj())


class TwoQubit:
    def __init__(self, q1, q2):
        """
            Initializes two qubits
        """
        self.q1 = q1
        self.q2 = q2

