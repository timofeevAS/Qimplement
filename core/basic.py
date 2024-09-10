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

'''
numpy vector describes Hadamard matrix 2 x 2:

see reference: https://en.wikipedia.org/wiki/Hadamard_matrix
'''
H = np.array([
    [1, 1],
    [1, -1]
], dtype=complex) / np.sqrt(2)

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