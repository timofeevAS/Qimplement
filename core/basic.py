import numpy as np

'''
numpy vector describes ket zero: |0>
'''
KET_0 = np.array([
    [1],
    [0]
], dtype=complex)

KET_1 = np.array([
    [0],
    [1]
], dtype=complex)

KET_00 = np.kron(KET_0, KET_0) # |00>
KET_01 = np.kron(KET_0, KET_1) # |00>
KET_10 = np.kron(KET_1, KET_0) # |00>
KET_11 = np.kron(KET_1, KET_1) # |00>

P_0 = np.array([[1, 0], [0, 0]])  # |0><0|
P_1 = np.array([[0, 0], [0, 1]])  # |1><1|

'''
numpy vector describes Hadamard matrix 2 x 2:

see reference: https://en.wikipedia.org/wiki/Hadamard_matrix
'''
H = np.array([
    [1, 1],
    [1, -1]
], dtype=complex) / np.sqrt(2)

H2 = np.kron(H, H)

def HN(N: int):
    H_N = H.copy()
    for i in range(N - 1):
        H_N = np.kron(H_N, H)
    return H_N

X = np.array([[0, 1], [1, 0]])

KET_PLUS = (KET_0 + KET_1) / np.sqrt(2)

KET_MINUS = (KET_0 - KET_1) / np.sqrt(2)

# Pauli's matrixes

PAULI_X = np.array([[0, 1], [1, 0]])

PAULI_Y = np.array([[0, -1j], [1j, 0]])

PAULI_Z = np.array([[1, 0], [0, -1]])

CNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]])

def RX(angle):
        rotation_matrix = np.array([[np.cos(angle / 2), -1j * np.sin(angle / 2)],
                                    [-1j * np.sin(angle / 2), np.cos(angle / 2)]])
        return rotation_matrix

# Bell states
# Функция для вычисления состояния Белла
def bell_state(state_type='phi_plus'):
    # Kroneker product for get bell states

    if state_type == 'phi_plus':
        return (1/np.sqrt(2)) * (KET_00 + KET_11)  # |Φ+⟩
    elif state_type == 'phi_minus':
        return (1/np.sqrt(2)) * (KET_00 - KET_11)  # |Φ-⟩
    elif state_type == 'psi_plus':
        return (1/np.sqrt(2)) * (KET_01 + KET_10)  # |Ψ+⟩
    elif state_type == 'psi_minus':
        return (1/np.sqrt(2)) * (KET_01 - KET_10)  # |Ψ-⟩
    else:
        raise ValueError("Invalid Bell state type. Choose from 'phi_plus', 'phi_minus', 'psi_plus', 'psi_minus'.")
