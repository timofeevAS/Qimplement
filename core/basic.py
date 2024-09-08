import numpy as np

'''
numpy vector describes ket zero: |0>
'''
KET_0 = np.array([
    [1],
    [0]
], dtype=complex)

'''
numpy vector describes Hadamard matrix 2 x 2:

see reference: https://en.wikipedia.org/wiki/Hadamard_matrix
'''
H = np.array([
    [1, 1],
    [1, -1]
], dtype=complex) / np.sqrt(2)