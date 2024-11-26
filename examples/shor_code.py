import random

import numpy as np

from core.basic import RX, CNOT, I, H, TOFFOLI, X, Z
from core.simulator import NQubitSimulator


def shor9():
    # Prepare system:
    # 9 qubits. (One of them target PHI)
    sim = NQubitSimulator(9)
    sim.apply_single_qubit_gate(RX(np.pi/5), 0)

    print('Initial system state:')
    # Print qubit PHI state
    for i in range(sim.dimension):
        print(f'State {i}: {sim.get_qubit_state(i)}')


    # Coding.
    # S1
    CNOT_03 = CNOT(9,0,3)
    sim.apply_n_qubit_gate(CNOT_03)

    # S2
    CNOT_06 = CNOT(9, 0, 6)
    sim.apply_n_qubit_gate(CNOT_06)

    # S3
    sim.apply_n_gates(H, I, I, H, I, I, H, I, I)

    # S4
    CNOT_01 = CNOT(N=9, c=0, t=1)
    CNOT_34 = CNOT(N=9, c=3, t=4)
    CNOT_67 = CNOT(N=9, c=6, t=7)
    S4_operator = np.dot(np.dot(CNOT_01, CNOT_34), CNOT_67)
    sim.apply_n_qubit_gate(S4_operator)

    # S5
    CNOT_02 = CNOT(N=9, c=0, t=2)
    CNOT_35 = CNOT(N=9, c=3, t=5)
    CNOT_68 = CNOT(N=9, c=6, t=8)
    S5_operator = np.dot(np.dot(CNOT_02, CNOT_35), CNOT_68)
    sim.apply_n_qubit_gate(S5_operator)

    print('Finishing coding part:')
    for i in range(sim.dimension):
        print(f'State {i}: {sim.get_qubit_state(i)}')


    # Random inverse qubit with PAULI [X or Z] (Simulate Error)
    idx = random.randint(0, 8) # Pick random index.
    pauli_x_error = random.random() > 0.5
    if pauli_x_error:
        sim.apply_single_qubit_gate(X, idx)
    else:
        sim.apply_single_qubit_gate(Z, idx)

    print('Finishing error simulation part:')
    for i in range(sim.dimension):
        print(f'State {i}: {sim.get_qubit_state(i)}')

    # Decoding
    # S6 - clone of S4
    sim.apply_n_qubit_gate(S4_operator)

    # S7 - clone of S5
    sim.apply_n_qubit_gate(S5_operator)

    # S8
    TOFFOLI_120 = TOFFOLI(N=9, controls=[1, 2], target=0)
    TOFFOLI_453 = TOFFOLI(N=9, controls=[4, 5], target=3)
    TOFFOLI_876 = TOFFOLI(N=9, controls=[8, 7], target=6)

    # Комбинируем все Toffoli-гейты в один оператор
    S8_operator = np.dot(np.dot(TOFFOLI_120, TOFFOLI_453), TOFFOLI_876)
    sim.apply_n_qubit_gate(S8_operator)

    # S9 - clone of S3
    sim.apply_n_gates(H, I, I, H, I, I, H, I, I)

    # S10 - clone of S1
    sim.apply_n_qubit_gate(CNOT_03)

    # S11 - clone of S2
    sim.apply_n_qubit_gate(CNOT_06)

    # S12
    TOFFOLI_360 = TOFFOLI(N=9, controls=[3,6], target=0)
    sim.apply_n_qubit_gate(TOFFOLI_360)

    print('Finishing decoding part')
    for i in range(sim.dimension):
        print(f'State {i}: {sim.get_qubit_state(i)}')


if __name__ == '__main__':
    shor9()

