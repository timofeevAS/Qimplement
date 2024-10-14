from typing import List

import numpy as np

from core.basic import X, HN, PAULI_X, KET_0, KET_1, H, CNOT, PAULI_Z, RX
from core.oracle import generate_oracle_simon
from core.simulator import NQubitSimulator

def teleport(sim: NQubitSimulator):
    """
    Quantum teleporation
    Circuit: https://vk.cc/cCxz9H
    Circuit with steps: https://yapx.ru/album/YEUQV
    """
    # Step 0
    sim.reset()  # |0...0>
    # Number of qubits = 3

    # Step 0 optional (rotate with Rx)
    sim.apply_single_qubit_gate(RX(np.pi/4), 0)

    # Step 1
    sim.apply_single_qubit_gate(H, 1) # Step 1 Hadamard gate for 2nd (1) qubit.

    # Step 2
    sim.apply_n_gates(np.eye(2), CNOT) # Step 2

    # Step 3
    # Print 'states'
    print('Begin...')
    for i in range(sim.dimension):
        print(f'State {i}: {sim.get_qubit_state(i)}')

    # Apply Step 3 gate
    sim.apply_n_gates(CNOT, np.eye(2))

    # Step 4
    sim.apply_n_gates(H, np.eye(2), np.eye(2))

    # Step 5: Measure qubits 0 and 1
    measurement_results = sim.measure_multiple_qubits([0, 1])
    print(f'Measurements: {measurement_results}')
    print('Teleport...')
    # Step 6: Apply controlled gates based on the measurement results
    # If qubit 1 was measured as 1, apply X gate to qubit 2
    sim.controlled_by_measurement(np.eye(2), PAULI_X, measurement_results[1], 2)

    # If qubit 0 was measured as 1, apply Z gate to qubit 2
    sim.controlled_by_measurement(np.eye(2), PAULI_Z, measurement_results[0], 2)
    print('Result...')
    # Final state of the third qubit should be the teleported state
    for i in range(sim.dimension):
        print(f'State {i}: {sim.get_qubit_state(i)}')

if __name__ == '__main__':
    sim = NQubitSimulator(3)
    teleport(sim)






