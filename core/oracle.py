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
