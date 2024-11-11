import numpy as np

from core.basic import CNOT
from core.simulator import SimulatedQubit


def apply_cnot(control_bit: SimulatedQubit, target_bit: SimulatedQubit) -> SimulatedQubit:
    new_state = np.dot(CNOT, np.kron(control_bit.state, target_bit.state))
    new_qubit = SimulatedQubit()
    new_qubit.state = new_state
    return new_qubit


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