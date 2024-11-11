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

CNOT_matr = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]])

def RX(angle):
        rotation_matrix = np.array([[np.cos(angle / 2), -1j * np.sin(angle / 2)],
                                    [-1j * np.sin(angle / 2), np.cos(angle / 2)]])
        return rotation_matrix

# Bell states
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

def CNOT(N, c, t):
    if c >= t:
        raise ValueError("CNOT generator is correct only with c < t.")
    # based on:
    # https://quantumcomputing.stackexchange.com/questions/4252/how-to-derive-the-cnot-matrix-for-a-3-qubit-system-where-the-control-target-qu
    # AND
    # https://quantumcomputing.stackexchange.com/questions/4078/how-to-construct-a-multi-qubit-controlled-z-from-elementary-gates

    # I ⊗c
    I_c = np.eye(2**c)

    # |0><0| и |1><1|
    zero_projector = P_0
    one_projector = P_1

    # X
    X_gate = PAULI_X

    # I ⊗(t-c-1)
    I_tc = np.eye(2**(t-c-1))

    # I ⊗(n-t-1)
    I_nt = np.eye(2**(N-t-1))

    # I^⊗c ⊗ |0><0| ⊗ I^(n-c-1)
    term1 = np.kron(np.kron(I_c, zero_projector), np.eye(2**(N-c-1)))

    # I^⊗c ⊗ |1><1| ⊗ I^⊗(t-c-1) ⊗ X ⊗ I^⊗(n-t-1)
    term2 = np.kron(np.kron(np.kron(np.kron(I_c, one_projector), I_tc), X_gate), I_nt)

    # CNOT = term1 + term2
    CNOT_matrix = term1 + term2

    return CNOT_matrix
