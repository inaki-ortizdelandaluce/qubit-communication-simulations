# The clasical cost of transmitting a qubit 
Classical and quantum simulations of qubit transmission. Final Project for Postgraduate Degree in Quantum Engineering by Universitat Politècnica de Catalunya (UPC).

# Reference Paper
[The classical cost of transmitting a qubit - Renner, Tavakoli and Quintino, 2022 (arXiv2207.02244)](https://arxiv.org/abs/2207.02244)

# Objective
Prove by classical and quantum computer based experiments that a qubit transmission can be simulated classically with a total cost of 2 qubits for any general measurement, either in a prepare-and-measure or a Bell scenario.

# Roadmap

## Phase I - Simulate Prepare-and-Measure PVMs classically 
Deadline: 07/02 (after Quantum Computing sessions are completed)

### I.a - Generation of random states and projection-valued measurements

### I.b - Classical simulation using Python

### I.c - Generation of multiple-qubit states towards Bell scenarios preparation (see Phase IV)
---
## Phase II - Simulate Prepare-and-Measure PVMs with a quantum computer
Deadline: 23/02 (before next tutored session)

### II.a - Simulation using Qiskit Quantum Simulator

### II.b - Simulation using IBM Quantum Computer. Error correction 
---
## Phase III - Simulate Prepare-and-Measure with POVMs

### III.a - Generation of POVM measurements

### III.b - Perform classical and quantum computer simulations (as per Phases I and II)
---
## Phase IV - Simulate using Bell scenarios
---

# Tasks
Each of us will lead a different task but we will work simultaneously on any task and different aspects of the project if/when needed.
## Phase I

Aido:
- Classical simulation from Alice's perspective

Tomas:
- Classical simulation from Bob's perspective

Inaki: 
- Generation of random states and projective measurements
- Overall code integration and software infrastructure


# Open Questions

- Using pure states or mixed states
- Random states with Qiskit or uniform distribution over the Bloch sphere
 
 # Other References
- [The Communication Cost of Simulating Bell Correlations - Toner and Bacon, 2003](https://arxiv.org/abs/quant-ph/0304076)
- [Qiskit's Random State Vector API](http://qiskit.org/documentation/stubs/qiskit.quantum_info.random_statevector.html)
- Naimark's extension
