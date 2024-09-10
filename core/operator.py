import numpy as np

from core.basic import CNOT
from core.simulator import SimulatedQubit


def apply_cnot(control_bit: SimulatedQubit, target_bit: SimulatedQubit) -> SimulatedQubit:
    new_state = np.dot(CNOT, np.kron(control_bit.state, target_bit.state))
    new_qubit = SimulatedQubit()
    new_qubit.state = new_state
    return new_qubit