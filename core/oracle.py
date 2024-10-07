import numpy as np
import itertools


def generate_oracle_deutsch_jozsa(N, boolean_function):
    """
    Generate Oracle matrix for {0,1}^n -> {0,1}

    Args:
    - N: Number of variables.
    - boolean_function: function

    Returns:
    - Matrix of 2^(N+1) x 2^(N+1), describes Oracle.
    """
    # Dimension of Oracle
    matrix_size = 2 ** (N + 1)

    # Init zero matrix
    oracle_matrix = np.zeros((matrix_size, matrix_size), dtype=int)

    # Calculate each values
    for x in itertools.product([0, 1], repeat=N):
        for y in [0, 1]:  # y - extra qubit which apply f(x)
            input_index = int(''.join(map(str, x)) + str(y), 2)

            # Value
            f_x = boolean_function(x)
            output_y = y ^ f_x  # y XOR f(x)

            # Output index
            output_index = int(''.join(map(str, x)) + str(output_y), 2)

            # Set 1 in INPUT, OUTPUT position
            oracle_matrix[input_index][output_index] = 1

    return oracle_matrix

import numpy as np
import itertools

def generate_oracle_bernstein_vazirani(N, s):
    """
    Generate Oracle matrix for Bernstein-Vazirani problem.

    Args:
    - N: Number of variables.
    - s: Hidden string used for scalar product in the boolean function.

    Returns:
    - Matrix of 2^(N+1) x 2^(N+1), describes Oracle.
    """
    # Dimension of Oracle matrix
    matrix_size = 2 ** (N + 1)

    # Init zero matrix
    oracle_matrix = np.zeros((matrix_size, matrix_size), dtype=int)

    # Iterate over all possible inputs (x) and auxiliary bit (y)
    for x in itertools.product([0, 1], repeat=N):
        for y in [0, 1]:  # y - auxiliary qubit
            # Input index (binary representation of x followed by y)
            input_index = int(''.join(map(str, x)) + str(y), 2)

            # Calculate the boolean function f(x) = s · x mod 2 (scalar product)
            f_x = np.dot(s, x) % 2

            # Calculate the output value for the auxiliary bit
            output_y = y ^ f_x  # y XOR f(x)

            # Output index (binary representation of x followed by new y)
            output_index = int(''.join(map(str, x)) + str(output_y), 2)

            # Set the matrix element at (input_index, output_index) to 1
            oracle_matrix[input_index][output_index] = 1

    return oracle_matrix

if __name__ == '__main__':
    # Example
    def xor(x):
        # XOR function
        return x[0] ^ x[1]

    def const_zero(x):
        return 0

    N = 1
    oracle = generate_oracle_deutsch_jozsa(1, const_zero)
    print(np.array(oracle))

    N = 2
    oracle = generate_oracle_deutsch_jozsa(N, xor)
    print(np.array(oracle))

    # Example Bernstein-Vazirani
    # Example: hidden string s = [1, 0] (N = 2)
    print('Bernstein Vazirani:')
    s = [1, 0]
    N = len(s)

    oracle = generate_oracle_bernstein_vazirani(N, s)
    print(np.array(oracle))

