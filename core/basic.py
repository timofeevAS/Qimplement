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