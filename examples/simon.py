from collections import defaultdict
from typing import List

import numpy as np

from core.basic import X, HN, PAULI_X, KET_0, KET_1, CNOT
from core.oracle import generate_oracle_simon
from core.simulator import NQubitSimulator

def simon(sim: NQubitSimulator, oracle) -> List[bool]:
    """
    Simon-problem quantum circuit:
    |0> -- ~ -- H -- ORACLE -- H -- M
    ...
    N
    |0> -- ~ -- H -- ORACLE -- H -- M
    ...
    |0> -- X -- H -- ORACLE -- ~ -- ...
    ...
    |0> -- X -- H -- ORACLE -- ~ -- ...
    .......|....|........|....|....|
    .......1....2........3....4....5

    By measured first N qubit after STEP 3 and STEP 4 we get binary digit s:
    answer to question:
    f: {0,1}^n → {0,1}^n;
    ∃! s = 0: ∀ x f(x) = f(y) ⇐⇒ y = x ⊕ s

    s - ?
    """
    sim.reset()  # |0...0>

    # prepare gate H x H x H x H x I x I x I x I
    hadamard_step1_gate = HN(sim.dimension//2)
    for _ in range(sim.dimension//2):
        hadamard_step1_gate = np.kron(hadamard_step1_gate, np.eye(2))

    sim.apply_n_qubit_gate(hadamard_step1_gate)  # Step 2
    sim.apply_n_qubit_gate(oracle)  # Step 3
    sim.apply_n_qubit_gate(hadamard_step1_gate)


    measured = sim.measure_multiple_qubits(list(range(sim.dimension//2)))
    return measured

def example_n2_s11():
    # Example
    N = 2  # Len of s = '11'
    measured_y = set()

    # Oracle for s = 11. Taken from:
    # https://github.com/Qiskit/textbook/blob/main/notebooks/ch-algorithms/simon.ipynb.

    oracle = CNOT(4, 0, 2) @ CNOT(4, 0, 3) @ CNOT(4, 1, 2) @ CNOT(4, 1, 3)
    ITER_COUNT = 1024
    m = defaultdict(int)

    for i in range(ITER_COUNT):
        sim = NQubitSimulator(N * 2)
        result = simon(sim, oracle)
        if result == [0, 0]:
            continue
        measured_y.add(''.join(map(str, result)))  # Convert list to str ex: [1, 0, 1] -> '101'
        m[''.join(map(str, result))] += 1

    print(measured_y)
    print(m)
    return m['11']


def example_n3_s100():
    # Example
    N = 3  # Len of s = '100'
    measured_y = set()

    # Oracle for s = 100. Taken from:
    # https://quantum-ods.github.io/qmlcourse/book/qcalgo/ru/simon_algorithm.html

    oracle = CNOT(6, 0, 3)
    ITER_COUNT = 1024
    m = defaultdict(int)

    for i in range(ITER_COUNT):
        sim = NQubitSimulator(N * 2)
        result = simon(sim, oracle)
        if result == [0, 0]:
            continue
        measured_y.add(''.join(map(str, result)))  # Convert list to str ex: [1, 0, 1] -> '101'
        m[''.join(map(str, result))] += 1

    print(measured_y)
    print(m)
    return m['100']


if __name__ == '__main__':
    example_n2_s11()
    example_n3_s100()





