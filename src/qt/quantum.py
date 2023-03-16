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

    job = backend.run(qc_transpiled, shots=shots)
    result = job.result()
    counts = result.get_counts(qc_transpiled)

    return counts
