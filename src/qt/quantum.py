import numpy as np
from qiskit import QuantumCircuit, transpile, Aer, IBMQ
from qt.qubit import Qubit
from qt.measurement import POVM


def prepare_and_measure_povm(shots, qubit: Qubit, povm: POVM):
    # TODO extend usage to any POVM of N elements (currently N =4)
    qc = QuantumCircuit(2, 2)
    u = povm.unitary()

    qc.initialize(qubit.ket(), 0)
    qc.unitary(u, [0, 1])
    qc.measure([0, 1], [0, 1])

    backend = Aer.get_backend('aer_simulator')
    qc_transpiled = transpile(qc, backend)

    job = backend.run(qc_transpiled, shots=shots, memory=True)
    result = job.result()
    counts = result.get_counts(qc_transpiled)

    p = np.array([counts['00'], counts['01'], counts['10'], counts['11']])
    p = p / np.sum(p)

    results = {
        "counts": counts,
        "memory": result.get_memory(),
        "probabilities": p
    }
    return results
