import numpy as np
import itertools

from core.basic import H, KET_0, PAULI_X, PAULI_Y, PAULI_Z, CNOT, P_0, P_1
from core.interface import QubitInterface, QuantumDevice

def qft(n: int):
    """
    Generate the QFT matrix for n qubits.

    Args:
    - n (int): The number of qubits.

    Returns:
    - QFT matrix (numpy array)
    """
    N = 2 ** n
    qft_matrix = np.zeros((N, N), dtype=complex)

    omega = np.exp(2j * np.pi / N)
    for j in range(N):
        for k in range(N):
            qft_matrix[j, k] = omega ** (j * k)

    # Normalize the matrix by dividing by sqrt(N)
    qft_matrix /= np.sqrt(N)
    return qft_matrix


def qft_dagger(n: int):
    """
    Generate the inverse QFT (QFT†) matrix for n qubits.

    Args:
    - n (int): The number of qubits.

    Returns:
    - QFT† matrix (numpy array)
    """
    # Получаем обычную QFT и берём её комплексно сопряжённое и транспонированное
    qft_matrix = qft(n)
    qft_dagger_matrix = np.conjugate(qft_matrix).T
    return qft_dagger_matrix

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

class NQubitSimulator:
    def __init__(self, n: int):
        available_qubits = [SimulatedQubit() for _ in range(n)]
        self.state = KET_0
        self.dimension = n
        self.reset() # Call reset() to set |0...0>
        self.collapsed = [None for _ in range(n)] # Collapsed qubits.

    def apply_single_qubit_gate(self, gate, qubit_idx: int):
        operation = None
        if qubit_idx == 0:
            # Generate operation matrix: GATE x I x I x ... x I
            operation = gate
            for i in range(1, self.dimension):
                operation = np.kron(operation, np.eye(2))
        elif qubit_idx == self.dimension - 1:
            # Generate operation matrix: I x I x ... x G x I x ... x I
            operation = np.eye(2)
            for i in range(0, self.dimension - 2):
                operation = np.kron(operation, np.eye(2))
            operation = np.kron(operation, gate)
        elif 0 < qubit_idx < self.dimension - 1:
            # Generate operation matrix: I x I x ...x I x G
            operation = np.eye(2)
            for i in range(0, qubit_idx - 1):
                operation = np.kron(operation, np.eye(2))
            operation = np.kron(operation, gate)
            for i in range(qubit_idx + 1, self.dimension):
                operation = np.kron(operation, np.eye(2))
        else:
            raise ValueError(f"Invalid qubit index. Only in range [0;{self.dimension}).")

        # Apply operation to quantum state.
        self.state = operation @ self.state

    def apply_n_qubit_gate(self, gate):
        # TODO: Is it need to check dimension?
        self.state = gate @ self.state

    def measure(self, qubit_idx: int) -> bool:
        """
        Measure the state of the specified qubit.
        """
        if qubit_idx > self.dimension or qubit_idx < 0:
            raise ValueError(f"Invalid qubit index. Only in range [0;{self.dimension}).")

        # We'll just measure individual qubits by marginalizing over the other qubit.
        prob_zero = 0
        combinations = list(itertools.product([0, 1], repeat=self.dimension))

        for i in range(len(combinations)):
            if combinations[i][qubit_idx] == 0:
                prob_zero += np.abs(self.state[i, 0]) ** 2

        is_measured_zero = np.random.random() <= prob_zero
        return bool(0 if is_measured_zero else 1)

    def single_measure(self, qubit_idx: int) -> bool:
        if qubit_idx >= self.dimension or qubit_idx < 0:
            raise ValueError(f"Invalid qubit index. Only in range [0;{self.dimension}).")

        if self.collapsed[qubit_idx] is not None:
            # TODO: Is it correct to return classic bit?
            return self.collapsed[qubit_idx]

        operator_0 = np.eye(1)
        operator_1 = np.eye(1)

        for i in range(self.dimension):
            if i == qubit_idx:
                operator_0 = np.kron(operator_0, P_0)
                operator_1 = np.kron(operator_1, P_1)
            else:
                operator_0 = np.kron(operator_0, np.eye(2))
                operator_1 = np.kron(operator_1, np.eye(2))

        projected_state_0 = operator_0 @ self.state
        probability_0 = np.abs(np.vdot(projected_state_0, projected_state_0))

        is_measured_zero = np.random.random() <= probability_0

        if is_measured_zero:
            self.state = projected_state_0 / np.sqrt(probability_0)
            self.collapsed[qubit_idx] = False
            return False
        else:
            projected_state_1 = operator_1 @ self.state
            probability_1 = np.abs(np.vdot(projected_state_1, projected_state_1))
            self.state = projected_state_1 / np.sqrt(probability_1)
            self.collapsed[qubit_idx] = True
            return True

    def measure_multiple_qubits(self, qubit_indices: list) -> list:
        """
        Measure a couple of qubits for one step.
        :param qubit_indices: List of qubits' indices
        :return: result
        """
        for qubit_idx in qubit_indices:
            if qubit_idx >= self.dimension or qubit_idx < 0:
                raise ValueError(f"Invalid qubit index: {qubit_idx}. Only in range [0; {self.dimension}).")

        # All posible qubit variations: |000...000> |000...001> ... |111...111>
        possible_outcomes = list(itertools.product([0, 1], repeat=len(qubit_indices)))

        probabilities = []
        projectors = []

        for outcome in possible_outcomes:
            operator = np.eye(1)
            outcome_projectors = []

            # Generate porjector operator;
            for idx, qubit_idx in enumerate(qubit_indices):
                projector = P_0 if outcome[idx] == 0 else P_1
                outcome_projectors.append(projector)

            # Kroneker product for each operator
            for i in range(self.dimension):
                if i in qubit_indices:
                    idx_in_measure = qubit_indices.index(i)
                    operator = np.kron(operator, outcome_projectors[idx_in_measure])
                else:
                    operator = np.kron(operator, np.eye(2))

            # Apply projector opertator to system.
            projected_state = operator @ self.state
            probability = np.abs(np.vdot(projected_state, projected_state))

            # Save probability for current projector
            probabilities.append(probability)
            projectors.append(operator)

        # Normaplize probabilities.
        total_probability = sum(probabilities)
        probabilities = [p / total_probability for p in probabilities]

        random_value = np.random.random()
        cumulative_probability = 0
        measured_outcome = None

        # Measure results.
        for idx, prob in enumerate(probabilities):
            cumulative_probability += prob
            if random_value <= cumulative_probability:
                measured_outcome = possible_outcomes[idx]
                break

        # Collapse
        projector = projectors[possible_outcomes.index(measured_outcome)]
        self.state = projector @ self.state / np.sqrt(probabilities[possible_outcomes.index(measured_outcome)])
        for idx, qubit_idx in enumerate(qubit_indices):
            self.collapsed[qubit_idx] = measured_outcome[idx]

        return list(measured_outcome)

    def reset(self):
        self.state = KET_0
        for i in range(1, self.dimension):
            self.state = np.kron(self.state, KET_0)

    def apply_n_gates(self, *gates):
        # Start with the first gate
        operation = gates[0]

        # Perform Kronecker product for each gate, if no gate provided for a specific qubit, use identity matrix
        for i in range(1, len(gates)):
            gate = gates[i]
            if gate is None:
                gate = np.eye(2)  # Identity gate for qubits that don't have a specific gate
            operation = np.kron(operation, gate)

        # Apply the final operation to the quantum state
        self.state = operation @ self.state

    def get_qubit_state(self, idx: int):
        if idx < 0 or idx >= self.dimension:
            raise ValueError(f"Invalid qubit index. Must be in range [0; {self.dimension}).")
        projector_0 = np.eye(1)
        projector_1 = np.eye(1)

        for i in range(self.dimension):
            if i == idx:
                projector_0 = np.kron(projector_0, P_0)
                projector_1 = np.kron(projector_1, P_1)
            else:
                projector_0 = np.kron(projector_0, np.eye(2))
                projector_1 = np.kron(projector_1, np.eye(2))

        projected_state_0 = projector_0 @ self.state
        projected_state_1 = projector_1 @ self.state

        prob_0 = np.abs(np.vdot(projected_state_0, projected_state_0))
        prob_1 = np.abs(np.vdot(projected_state_1, projected_state_1))

        return {'|0>': prob_0, '|1>': prob_1}

    def controlled_by_measurement(self, gate_if_0, gate_if_1, measured_value, target_qubit_idx: int):
        """
        Apply a gate to the target qubit, based on the result of a measured value.
        :param gate_if_0: The gate to apply if the measured value is 0.
        :param gate_if_1: The gate to apply if the measured value is 1.
        :param measured_value: The result of the measurement (0 or 1).
        :param target_qubit_idx: The index of the qubit to which the gate is applied.
        """
        if measured_value == 0:
            self.apply_single_qubit_gate(gate_if_0, target_qubit_idx)
        elif measured_value == 1:
            self.apply_single_qubit_gate(gate_if_1, target_qubit_idx)
        else:
            raise ValueError(f"Invalid measured value {measured_value}. Must be 0 or 1.")

    def apply_qft(self):
        """
        Apply the Quantum Fourier Transform (QFT) to the entire system.
        """
        qft_matrix = qft(self.dimension)  # create QFT for the current number of qubits
        self.state = qft_matrix @ self.state

    def apply_qft_dagger(self):
        """
        Apply the inverse Quantum Fourier Transform (QFT†) to the entire system.
        """
        qft_dagger_matrix = qft_dagger(self.dimension)  # create QFT† for the current number of qubits
        self.state = qft_dagger_matrix @ self.state