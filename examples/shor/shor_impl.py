import random
from typing import List
import numpy as np

from math import gcd, sqrt, ceil, log

from core.basic import *
from core.simulator import *

def tensormany(*entities):
    res = 1
    for entity in entities:
        res = np.kron(res, entity)
    return res

def gen_c_operator(control, target, operation):
    I = np.eye(2)
    proj_basis = KET_1

    size = abs(control - target) + 1

    c_operator = 1
    c_operator_complement = 1

    for i in range(size):

        complement_operator = I
        operator = I

        if i == control:
            operator = proj_basis @ proj_basis.transpose()
            complement_operator = (X @ proj_basis) @ (X @ proj_basis).transpose()

        if i == target:
            operator = operation

        c_operator = tensormany(c_operator, operator)
        c_operator_complement = tensormany(c_operator_complement, complement_operator)

    c_operator += c_operator_complement

    return c_operator

def gen_swap(size):
    operator = gen_c_operator(0, size - 1, X)
    operator = operator @ gen_c_operator(size - 1, 0, X)
    operator = operator @ gen_c_operator(0, size - 1, X)

    return operator

def a_mod_21(a: int):
    I = np.eye(2)
    
    if a == 2:
        operator = tensormany(KET_0 @ KET_0.transpose(), I, I, I, I) + \
                   tensormany(KET_1 @ KET_1.transpose(), I, I, gen_swap(2))

        operator = operator @ \
                   (tensormany(KET_0 @ KET_0.transpose(), I, I, I, I) +
                    tensormany(KET_1 @ KET_1.transpose(), gen_swap(2), I, I))

        operator = operator @ tensormany(I, I, gen_c_operator(2, 0, X)) @ gen_c_operator(4, 0, X)
        operator = operator @ tensormany(I, I, I, gen_swap(2))
        operator = operator @ tensormany(gen_swap(4), I)
        operator = operator @ tensormany(I, I, gen_swap(2), I)
        operator = operator @ tensormany(I, gen_swap(2), I, I)

        return operator
    elif a == 4:
        return gen_swap(5) @ tensormany(I, I, gen_swap(3))
    elif a == 5:
        return tensormany(X, I, X, I, X) @ tensormany(I, I, gen_swap(3)) @ gen_swap(5)
    elif a == 8:
        return tensormany(gen_swap(4), I)
    elif a == 13:
        return tensormany(I, I, X, I, X)
    elif a == 16:
        return tensormany(I, I, gen_swap(3)) @ gen_swap(5)
    elif a == 17:
        return tensormany(X, I, X, I, X) @ tensormany(I, I, gen_swap(3)) @ tensormany(gen_swap(3), I, I)
    elif a == 20:
        return tensormany(X, I, X, I, X)
    else:
        raise Exception(f"a: {a} not implement.")

def quantum_find_order(x):
    """
    Quantum Shor Algorithm
    Circuit: TODO...
    """
    N=7
    M=5
    I=np.eye(2)
    sim = NQubitSimulator(N+M)
    for i in range(N):
        sim.apply_single_qubit_gate(i, H)
    sim.apply_single_qubit_gate(X, N)
    oracle_matrix = a_mod_21(x)
    for i in range(N):
        mod = tensormany(*[I for _ in range(i)],
                            KET_1 @ KET_1.transpose(),
                            *[I for _ in range(N - i - 1)],
                            oracle_matrix)
        mod += tensormany(*[I for _ in range(i)],
                             KET_0 @ KET_0.transpose(),
                             *[I for _ in range(N - i + 5 - 1)])
        oracle_matrix = oracle_matrix @ oracle_matrix
        sim.apply_n_qubit_gate(oracle_matrix)

        qft_matrix = tensormany(qft_dagger(N), *[I for _ in range(M)])
        sim.apply_n_qubit_gate(qft_matrix)

        bits = sim.measure_multiple_qubits(list(range(N)))
        return bits

def calc_base(n):
    if n == 1:
        return 1
    for i in range(2, ceil(sqrt(n)) + 1):
        possible_base = log(n, i)
        if possible_base.is_integer():
            return i, round(possible_base)
    return None

def run_shor(n):
    if n % 2 == 0:
        print("n - even")
        return 2, n // 2

    if calc_base(n):
        base, power = calc_base(n)
        print(f"Provided number is {base} ^ {power}")
        return calc_base(n)[0]

    if not n == 21:
        raise Exception("This implementation doesn't factor this number yet")

    pick_pool = [i for i in range(2, 21)]
    while True:
        try:
            x = random.choice(pick_pool)
            pick_pool.remove(x)
            print(f"Randomly picked number {x}")

            gcd_value = gcd(x, n)
            print(f"gcd({x}, {n}) = {gcd_value}")

            if gcd_value > 1:
                return gcd_value, n // gcd_value

            print("Provided number will be factorised with quantum computation")

            r = quantum_find_order(x)
            print(f"Period for {x}^n mod 21 is {r}")
            if r % 2 != 0:
                raise Exception("Period value is odd, let's pick another number")

            guess1 = x ** (r // 2) - 1
            guess2 = x ** (r // 2) + 1
            if 21 in [gcd(guess1, n), gcd(guess2, n)]:
                raise Exception("One of the factors is 21, let's pick another number")
            return gcd(n, guess1), gcd(n, guess2)
        except Exception as e:
            print(str(e))

if __name__ == '__main__':
    impl = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 21]

    for digit in impl:
        try:
            print(f"Answer for {digit}: {'* '.join([str(i) for i in sorted(run_shor(digit))])}")
        except:
            print(f'Digit not impl: {digit}')





