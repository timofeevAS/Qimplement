import random
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np

from core.basic import RX, CNOT, I, H, TOFFOLI, X, Z, HN
from core.simulator import NQubitSimulator
import concurrent.futures

def z_correction(P=0.05, debug=False):
    # Prepare system:
    # 3 qubits. (One of them target PHI)
    sim = NQubitSimulator(3)
    sim.apply_single_qubit_gate(RX(np.pi / 5), 0)
    inital_state = sim.get_qubit_state_raw(0)

    if debug:
        print('Initial system state:')
        # Print qubit PHI state
        for i in range(sim.dimension):
            print(f'State {i}: {sim.get_qubit_state(i)}')

    # Coding.
    # S1
    CNOT_01 = CNOT(3, 0, 1)
    sim.apply_n_qubit_gate(CNOT_01)

    # S2
    CNOT_02 = CNOT(3, 0, 2)
    sim.apply_n_qubit_gate(CNOT_02)

    if debug:
        print('Finishing coding part:')
        for i in range(sim.dimension):
            print(f'State {i}: {sim.get_qubit_state(i)}')

    sim.apply_n_qubit_gate(HN(3))

    error_count = 0
    # Random inverse qubit with PAULI [Z] (Simulate Error)
    for idx in range(0, sim.dimension):
        if random.random() <= P:
            error_count += 1
            if debug:
                print(f'Error in {idx}')
            sim.apply_single_qubit_gate(Z, idx)
    if debug:
        print('Finishing error simulation part:')
        for i in range(sim.dimension):
            print(f'State {i}: {sim.get_qubit_state(i)}')

        if error_count:
            print(f'{error_count} - P: {P}')
    sim.apply_n_qubit_gate(HN(3))
    # Decoding
    sim.apply_n_qubit_gate(CNOT_01)
    sim.apply_n_qubit_gate(CNOT_02)

    TOFFOLI_120 = TOFFOLI(N=3, controls=[1, 2], target=0)

    sim.apply_n_qubit_gate(TOFFOLI_120)

    if debug:
        print('Finishing decoding part')
        for i in range(sim.dimension):
            print(f'State {i}: {sim.state[0]}')

    finite_state = sim.get_qubit_state_raw(0)
    if np.isclose(finite_state['|0>'], inital_state['|0>'], 0.01) and np.isclose(finite_state['|1>'],
                                                                                 inital_state['|1>'], 0.01):
        return (True, error_count)
    else:
        return (False, error_count)

def no_correction(P=0.05, debug = False):

    sim = NQubitSimulator(1)
    sim.apply_single_qubit_gate(RX(np.pi / 5), 0)
    inital_state = sim.get_qubit_state_raw(0)

    # No correction.
    # Error simulation.
    error_count = 0
    # Random inverse qubit with PAULI [Z] (Simulate Error)
    if random.random() <= P:
        error_count += 1
        if debug:
            print(f'Error applied')
        sim.apply_single_qubit_gate(Z, 0)

    # Check.
    finite_state = sim.get_qubit_state_raw(0)
    if np.isclose(finite_state['|0>'], inital_state['|0>'], 0.0001) and np.isclose(finite_state['|1>'],
                                                                                 inital_state['|1>'], 0.0001):
        return True, error_count
    else:
        return False, error_count

def theory(p):
    return 3 * p ** 2 - 2 * p ** 3


def compute_failure_probability(P, total_rounds=500):
    failure_count = 0
    failure_count2 = 0

    # Если P == 0, то вероятность ошибки = 0
    if P == 0:
        return (P, 0, 0)

    for i in range(total_rounds):
        correct = z_correction(P=P)  # True if error correction successful, False if failure.
        if not correct[0]:
            failure_count += 1

    p_e = failure_count / total_rounds  # Probability of error for total rounds
    p_e_nc = P  # Placeholder for second variable, modify as needed
    print(f'finished P={P}')
    return (P, p_e, p_e_nc)


def parallel_compute_failure_probability(p_values, total_rounds=500):
    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Теперь передаем саму функцию без lambda
        results = list(executor.map(compute_failure_probability, p_values, [total_rounds]*len(p_values)))
    return results

def z_correction_withplot():
    import matplotlib.pyplot as plt

    p = [0, 0.05, 0.1, 0.15, 0.20, 0.30, 0.4, 0.45, 0.50, 0.55, 0.60]
    p_e = []
    p_e_nc = []
    total_rounds = 10000

    results = parallel_compute_failure_probability(p, total_rounds)


    for result in results:
        P, p_e_value, p_e_nc_value = result
        p_e.append(p_e_value)
        p_e_nc.append(p_e_nc_value)

        print(f'finished {P} with P_e: {p_e_value} || P_e_nc: {p_e_nc_value}')

    plt.plot(p, p_e, marker='o',label='Исправление Z ошибки', linestyle='-', color='b', alpha=0.7)
    plt.plot(p, p_e_nc,label='Без исправления', marker='o', linestyle='-', color='r', alpha=0.7)

    p_values = np.linspace(0, max(p), 100)
    theory_values = theory(p_values)

    plt.plot(p_values, theory_values, label='Теоретическое значение', color='green', linestyle='--', linewidth=2.5)
    plt.xlabel('Вероятность ошибки P')  # Подпись оси X
    plt.ylabel('Общая вероятность ошибки P_e')  # Подпись оси Y
    plt.title('Зависимость вероятности ошибки от P')  # Заголовок графика
    plt.grid(True)  # Включение сетки
    plt.legend()
    plt.savefig('error_probability_graph.pdf', format='pdf')
    plt.show()  # Отображение графика


if __name__ == '__main__':
    z_correction_withplot()


