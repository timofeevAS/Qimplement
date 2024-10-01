import numpy as np

from core.basic import H2, H, KET_01
from core.simulator import TwoQubitSimulator

ORACLE1 = (
    np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]], dtype=complex))  # f(x) = 1

ORACLE2 = (
    np.array([
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]], dtype=complex))  # f(x) = 0

ORACLE3 = (
    np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]], dtype=complex))  # f(x) = x

ORACLE4 = (
    np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 0],
        [1, 0, 0, 0]], dtype=complex))  # f(x) = !x


def deutsch_algorithm(sim: TwoQubitSimulator, oracle) -> bool:
    """
    Deutsch algorithm quantum circuit
    |0> -- H -- ORACLE -- H -- M
    |1> -- H -- ORACLE -- ~ -- ...
    .......|.....|........|....|
    .......1.....2........3....4
    """

    sim.set_state(KET_01)  # |01>
    sim.apply_two_qubit_gate(H2)  # Step 1
    sim.apply_two_qubit_gate(oracle)  # Step 2
    sim.apply_single_qubit_gate(H, 0)  # Step 3
    return sim.measure(0)  # Step 4


if __name__ == '__main__':
    # Example of Deutsch algorithm
    sim = TwoQubitSimulator()

    print('Run ORACLE1 (f(x) = 0):')
    print(f'Result: {deutsch_algorithm(sim, ORACLE1)}')

    print('Run ORACLE2 (f(x) = 1):')
    print(f'Result: {deutsch_algorithm(sim, ORACLE2)}')

    print('Run ORACLE3 (f(x) = x):')
    print(f'Result: {deutsch_algorithm(sim, ORACLE3)}')

    print('Run ORACLE4 (f(x) = !x):')
    print(f'Result: {deutsch_algorithm(sim, ORACLE4)}')


