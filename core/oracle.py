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

def generate_oracle_simon(N, s):
    """
    Generate Oracle matrix for Simon's problem.

    Args:
    - N: Number of variables (length of binary string).
    - s: Hidden string used for the periodic function f(x) = f(x ⊕ s).

    Returns:
    - Matrix of 2^(N+1) x 2^(N+1), describes Oracle.
    """
    matrix_size = 2 ** (N + 1)
    oracle_matrix = np.zeros((matrix_size, matrix_size), dtype=int)

    # Storage visited x
    visited = set()

    # Iterate all x:
    for x in itertools.product([0, 1], repeat=N):
        x = np.array(x)
        x_tuple = tuple(x)

        if x_tuple in visited:
            continue

        # Calculate 2nd point: x ⊕ s
        x_xor_s = x ^ s
        x_xor_s_tuple = tuple(x_xor_s)

        # Mark both points as visited.
        visited.add(x_tuple)
        visited.add(x_xor_s_tuple)

        # f_x random value
        f_x = np.random.randint(2)

        for y in [0, 1]:
            # Input index for x and y
            input_index = int(''.join(map(str, x)) + str(y), 2)
            # Output index for x and y ⊕ f(x)
            output_index = int(''.join(map(str, x)) + str(y ^ f_x), 2)
            oracle_matrix[input_index][output_index] = 1

            # Input index for x ⊕ s and y
            input_index_xor_s = int(''.join(map(str, x_xor_s)) + str(y), 2)
            # Output index for x ⊕ s and y ⊕ f(x)
            output_index_xor_s = int(''.join(map(str, x_xor_s)) + str(y ^ f_x), 2)
            oracle_matrix[input_index_xor_s][output_index_xor_s] = 1

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

    # Example Simon problem
    # Example: hidden string s = [1, 0] (N = 2)
    print('Simon pronlem:')
    s = [1, 0]
    N = len(s)

    oracle = generate_oracle_simon(N, s)
    print(np.array(oracle))