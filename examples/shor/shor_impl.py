import random


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

def shor_oracle(x, n):
    """
    Oracle for x mod n:

    |y> -> |xy mod n>
    """

    n_bits = int(np.ceil(np.log2(n)))
    result = np.zeros(shape=(2 ** n_bits, 2 ** n_bits), dtype=complex)
    for y_i in range(2 ** n_bits):
        if y_i >= n:
            result[y_i, y_i] = 1
        else:
            result[y_i, (x * y_i) % n] = 1
    return result

def quantum_find_order(x):
    """
    Quantum Shor Algorithm
    Circuit: https://physlab.org/wp-content/uploads/2023/05/Shor_s_Algorithm_23100113_Fin.pdf
    """
    N=7
    M=5
    I=np.eye(2)
    sim = NQubitSimulator(N+M)
    for i in range(N):
        sim.apply_single_qubit_gate(i, H)
    sim.apply_single_qubit_gate(X, N)
    oracle_matrix = shor_oracle(x, 21)
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

    pool = list(range(2,n))
    while True:
        try:
            x = random.choice(pool)
            pool.remove(x)
            print(f"Popped number {x}")

            gcd_value = gcd(x, n)
            print(f"gcd({x}, {n}): {gcd_value}")

            if gcd_value > 1:
                return gcd_value, n // gcd_value

            print("Quantum processing.")

            a = quantum_find_order(x)
            print(f"Period {x}^n mod 21 is {a}")
            if a % 2 != 0:
                raise Exception("Period value is odd")

            guess1 = x ** (a // 2) - 1
            guess2 = x ** (a // 2) + 1
            return gcd(n, guess1), gcd(n, guess2)
        except Exception as e:
            print(str(e))

if __name__ == '__main__':
    DEBUG = False
    impl = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 21]
    results = []
    for digit in impl:
        answer=f'{digit}: {"* ".join([str(i) for i in sorted(run_shor(digit))])}'
        results.append(answer)
        try:
            print(answer)
        except:
            print(f'Digit not impl: {digit}')

    for res in results:
        print(res)





