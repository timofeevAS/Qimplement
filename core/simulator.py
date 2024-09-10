import numpy as np

from core.basic import H, KET_0, PAULI_X, PAULI_Y, PAULI_Z
from core.interface import QubitInterface, QuantumDevice


class SimulatedQubit(QubitInterface):
    def __init__(self):
        self.state = np.array([[0], [0]], dtype=complex)
        self.reset()

    def h(self):
        self.state = H @ self.state

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


class SingleQubitSimulator(QuantumDevice):
    available_qubits = [SimulatedQubit()]

    def allocate_qubit(self) -> SimulatedQubit:
        if self.available_qubits:
            return self.available_qubits.pop()

    def deallocate_qubit(self, qubit: SimulatedQubit):
        self.available_qubits.append(qubit)
