from typing import List

import numpy as np

from core.basic import X, HN, PAULI_X
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



if __name__ == '__main__':
    # Example
    N = 2 # Len of s
    s = '11'

    sim = NQubitSimulator(N * 2)
    oracle = generate_oracle_simon(N, s)
    result = simon(sim, oracle)

    print(f'Hidden binary period s: {s}')
    print(f'Measured: {result}')





