from typing import List

from core.basic import X, HN
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
    .......|....|........|....|....|
    .......1....2........3....4....5

    By measured first N qubit after STEP 3 and STEP 4 we get binary digit s:
    answer to question:
    f: {0,1}^n → {0,1}^n;
    ∃! s = 0: ∀ x f(x) = f(y) ⇐⇒ y = x ⊕ s

    s - ?
    """

    sim.reset()  # |0...0>
    sim.apply_single_qubit_gate(X, sim.dimension - 1)  # Step 1
    sim.apply_n_qubit_gate(HN(sim.dimension))  # Step 2
    sim.apply_n_qubit_gate(oracle)  # Step 3
    sim.apply_n_qubit_gate(HN(sim.dimension))

    measured = []
    for i in range(sim.dimension - 1):
        measured.append(sim.measure(i))
    return measured



if __name__ == '__main__':
    # Example
    N = 3 # Len of s
    s = [1, 0, 1]

    sim = NQubitSimulator(N + 1)
    oracle = generate_oracle_simon(N, s)
    result = simon(sim, oracle)

    print(f'Hidden binary period s: {s}')
    print(f'Measured: {result}')





