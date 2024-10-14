from typing import List

import numpy as np
from pyexpat.errors import messages

from core.basic import H2, H, KET_01, X, HN
from core.oracle import generate_oracle_deutsch_jozsa
from core.simulator import TwoQubitSimulator, NQubitSimulator
from examples.deutsch import ORACLE1, ORACLE2, ORACLE3, ORACLE4


def deutsch_jozsa(sim: NQubitSimulator, oracle) -> List[bool]:
    """
    Deutsch algorithm quantum circuit
    |0> -- ~ -- H -- ORACLE -- H -- M
    ...
    N
    ...
    |0> -- X -- H -- ORACLE -- ~ -- ...
    .......|....|........|....|....|
    .......1....2........3....4....5
    """

    sim.reset()  # |0...0>
    sim.apply_single_qubit_gate(X, sim.dimension - 1)  # Step 1
    sim.apply_n_qubit_gate(HN(sim.dimension))  # Step 2
    sim.apply_n_qubit_gate(oracle)  # Step 3
    sim.apply_n_qubit_gate(HN(sim.dimension))

    measured = sim.measure_multiple_qubits(list(range(sim.dimension - 1)))
    return measured


if __name__ == '__main__':
    # Example of Deutsch-Jozsa algorithm for N = 2 (Classic Deutsch algorithm)
    print("Simulate classic Deutsch algorithm...")
    sim = NQubitSimulator(2)
    print('Run ORACLE1 (f(x) = 0):')
    print(f'Result: {deutsch_jozsa(sim, ORACLE1)}')

    print('Run ORACLE2 (f(x) = 1):')
    print(f'Result: {deutsch_jozsa(sim, ORACLE2)}')

    print('Run ORACLE3 (f(x) = x):')
    print(f'Result: {deutsch_jozsa(sim, ORACLE3)}')

    print('Run ORACLE4 (f(x) = !x):')
    print(f'Result: {deutsch_jozsa(sim, ORACLE4)}')

    # Example of Deutsch-Jozsa algorithm for N = 3 with Oracle (x1 AND x2)
    def bool_and(x) -> bool:
        return x[0] and x[1]
    oracle = generate_oracle_deutsch_jozsa(2, bool_and)

    print('Run with next Oracle (AND boolean):')
    print(np.array(oracle))

    sim = NQubitSimulator(3)
    print(f'Result: {deutsch_jozsa(sim, oracle)}')


    # Example of Deutsch-Jozsa algorithm for N = 3 with Oracle (const ONE)
    def bool_true(x) -> bool:
        return True

    oracle = generate_oracle_deutsch_jozsa(2, bool_true)

    print('Run with next Oracle (CONST TRUE):')
    print(np.array(oracle))

    sim = NQubitSimulator(3)
    print(f'Result: {deutsch_jozsa(sim, oracle)}')





