import numpy as np

from core.basic import H, KET_0, PAULI_X, PAULI_Y, PAULI_Z, CNOT
from core.interface import QubitInterface, QuantumDevice


class SimulatedQubit(QubitInterface):
    def __init__(self):
        self.state = np.array([[0], [0]], dtype=complex)
        self.reset()

    def copy(self) -> "SimulatedQubit":
        q = SimulatedQubit()
        q.state = self.state
        return q



    def h(self):
        self.state = H @ self.state

    def x(self):
        self.state = PAULI_X @ self.state

    def measure(self) -> bool:
        probability_zero = np.abs(self.state[0, 0]) ** 2  # probability measure ZERO (false)
        is_measured_zero = np.random.random() <= probability_zero

        return bool(0 if is_measured_zero else 1)

    def pauli_x(self):
        self.state = PAULI_X @ self.state

    def pauli_y(self):
        self.state = PAULI_Y @ self.state

    def pauli_z(self):
        self.state = PAULI_Z @ self.state

    def rotate_x(self, angle):
        rotation_matrix = np.array([[np.cos(angle / 2), -1j * np.sin(angle / 2)],
                                    [-1j * np.sin(angle / 2), np.cos(angle / 2)]])
        self.state = rotation_matrix @ self.state

    def rotate_y(self, angle):
        rotation_matrix = np.array([[np.cos(angle / 2), -np.sin(angle / 2)],
                                    [np.sin(angle / 2), np.cos(angle / 2)]])
        self.state = rotation_matrix @ self.state

    def rotate_z(self, angle):
        rotation_matrix = np.array([[np.exp(-1j * angle / 2), 0],
                                    [0, np.exp(1j * angle / 2)]])
        self.state = rotation_matrix @ self.state

    def reset(self):
        self.state = KET_0.copy()

    def ket0_p(self) -> float:
        probability_zero = np.abs(self.state[0, 0]) ** 2
        return probability_zero

    def ket1_p(self) -> float:
        probability_one = np.abs(self.state[1, 0]) ** 2 # Or 1 - P(ket0)
        return probability_one



class SingleQubitSimulator(QuantumDevice):
    available_qubits = [SimulatedQubit()]

    def allocate_qubit(self) -> SimulatedQubit:
        if self.available_qubits:
            return self.available_qubits.pop()

    def deallocate_qubit(self, qubit: SimulatedQubit):
        self.available_qubits.append(qubit)

class TwoQubitSimulator(QuantumDevice):
    available_qubits = [SimulatedQubit(), SimulatedQubit()]
    state = np.kron(available_qubits[0].state, available_qubits[1].state)  # Tensor product of two qubit states

    def allocate_qubit(self) -> SimulatedQubit:
        if self.available_qubits:
            return self.available_qubits.pop()

    def deallocate_qubit(self, qubit: SimulatedQubit):
        self.available_qubits.append(qubit)

    def apply_single_qubit_gate(self, gate, qubit_idx: int):
        if qubit_idx == 0:
            identity = np.eye(2)
            operation = np.kron(gate, identity)  # Apply gate to first qubit.
        elif qubit_idx == 1:
            identity = np.eye(2)
            operation = np.kron(identity, gate)  # Apply gate to second qubit.
        else:
            raise ValueError("Invalid qubit index. Only 0 or 1 allowed.")

        # Apply operation to quantum state.
        self.state = operation @ self.state

    def apply_two_qubit_gate(self, gate):
        # TODO: Is it need to check dimension?
        self.state = gate @ self.state

    def measure(self, qubit_idx: int) -> bool:
        """
        Measure the state of the specified qubit.
        """
        # We'll just measure individual qubits by marginalizing over the other qubit.
        if qubit_idx == 0:
            # Probability of measure |00> or |01>.
            prob_zero = np.abs(self.state[0, 0]) ** 2 + np.abs(self.state[1, 0]) ** 2
        elif qubit_idx == 1:
            # Probability of measure |00> or |10>.
            prob_zero = np.abs(self.state[0, 0]) ** 2 + np.abs(self.state[2, 0]) ** 2
        else:
            raise ValueError("Invalid qubit index. Only 0 or 1 allowed.")

        is_measured_zero = np.random.random() <= prob_zero
        return bool(0 if is_measured_zero else 1)

    def cnot(self):
        self.apply_two_qubit_gate(CNOT)

    def reset(self):
        self.state = np.kron(KET_0, KET_0)  # Reset to |00> state

    def set_state(self, state):
        """
        Set the state of the two qubits manually.
        """
        self.state = state