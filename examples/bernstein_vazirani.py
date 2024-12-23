from typing import List

from core.basic import X, HN
from core.oracle import generate_oracle_bernstein_vazirani
from core.simulator import NQubitSimulator

def bernstein_vazirani(sim: NQubitSimulator, oracle) -> List[int]:
    """
    Bernstein-Vazirani quantum circuit:
    |0> -- ~ -- H -- ORACLE -- H -- M
    ...
    N
    |0> -- ~ -- H -- ORACLE -- H -- M
    ...
    |0> -- X -- H -- ORACLE -- ~ -- ...
    .......|....|........|....|....|
    .......1....2........3....4....5

    By measured first N qubit after STEP 3 and STEP 4 we get binary digit s:
    answer to question:
    f: {0,1}^n → {0,1};
    f(x) = s•x = x1s1 XOR x2s2 XOR ... XOE xNsN.

    s - ?
    """

    sim.reset()  # |0...0>
    sim.apply_single_qubit_gate(X, sim.dimension - 1)  # Step 1
    sim.apply_n_qubit_gate(HN(sim.dimension))  # Step 2
    sim.apply_n_qubit_gate(oracle)  # Step 3
    sim.apply_n_qubit_gate(HN(sim.dimension))

    measured = sim.measure_multiple_qubits(list(range(sim.dimension-1)))
    return measured



if __name__ == '__main__':
    # Example
    s = [1, 0, 1, 0]
    N = len(s)  # Len of s

    sim = NQubitSimulator(N + 1)
    oracle = generate_oracle_bernstein_vazirani(N, s)
    result = bernstein_vazirani(sim, oracle)

    print(f'Hidden binary vector s: {s}')
    print(f'Measured: {result}')





