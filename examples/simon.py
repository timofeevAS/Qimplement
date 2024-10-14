from typing import List

import numpy as np

from core.basic import X, HN, PAULI_X, KET_0, KET_1
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
    N = 3 # Len of s


    def gen_c_operator(control, target, operation, reverse=False):
        proj_basis = KET_1
        if reverse:
            proj_basis = KET_0

        size = abs(control - target) + 1

        c_operator = 1
        c_operator_complement = 1

        for i in range(size):

            complement_operator = np.eye(2)
            operator = np.eye(2)

            if i == control:
                operator = proj_basis @ proj_basis.transpose()
                complement_operator = (X @ proj_basis) @ (X @ proj_basis).transpose()

            if i == target:
                operator = operation

            c_operator = tensordot_arb(c_operator, operator)
            c_operator_complement = tensordot_arb(c_operator_complement, complement_operator)

        c_operator += c_operator_complement

        return c_operator

    def tensordot_arb(*q_entities):
        res_q_entity = 1
        for i in range(len(q_entities)):
            res_q_entity = np.kron(res_q_entity, q_entities[i])
        return res_q_entity


    oracle = tensordot_arb(gen_c_operator(0, 3, X), np.eye(2), np.eye(2))
    oracle = tensordot_arb(np.eye(2), np.eye(2), gen_c_operator(0, 3, X)) @ oracle

    measured_y = set()

    ITER_COUNT = N*5

    for i in range(ITER_COUNT):
        sim = NQubitSimulator(N * 2)
        result = simon(sim, oracle)
        measured_y.add(''.join(map(str,result))) # Convert list to str ex: [1, 0, 1] -> '101'

    measured_y.remove('000')
    # Print user-friendly SLQ:
    print(f'Present SLQ by running: {ITER_COUNT}')
    for y in measured_y:
        s_idxs = [f's{i}' for i in range(N)]
        res = ''
        for idx, bit in enumerate(y):
            if bit == '1':
                res += f'{s_idxs[idx]} + '

        res = res.rstrip(' + ') + ' = 0'
        print(res)






