from core.basic import *
from core.simulator import NQubitSimulator
import numpy as np
from functools import reduce

from examples.shor.shor_impl import tensormany


def grover(N, marked_item):
    def compose(*args):
        args = args[::-1]
        return reduce(np.matmul, args[1:], args[0])

    sim = NQubitSimulator(N)
    for i in range(N):
        sim.apply_single_qubit_gate(H, i)

    ket_0_n = np.zeros((2 ** N, 1))
    ket_0_n[0, 0] = 1
    phase_flip = 2 * np.outer(ket_0_n, ket_0_n) - np.eye(2 ** N)

    oracle = np.eye(2 ** N)
    oracle[marked_item, marked_item] = -1

    diffusion_op = compose(
        tensormany(*([H] * N)),
        phase_flip,
        tensormany(*([H] * N)),
    )

    iterations = int(np.floor(np.pi / 4 * np.sqrt(2 ** N)))
    for _ in range(iterations):
        sim.apply_n_qubit_gate(oracle)
        sim.apply_n_qubit_gate(diffusion_op)

    return sim.measure_multiple_qubits(list(range(N)))


if __name__ == '__main__':
    guessed = 14
    got = grover(5, 2)
    print(f"guessed {guessed}, got {got}")