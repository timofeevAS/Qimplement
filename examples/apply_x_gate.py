from core.interface import QuantumDevice
import math

from core.simulator import SimulatedQubit, SingleQubitSimulator


def apply_x_example(device: QuantumDevice):
    # Quantum circuit:
    # |0> -->[Ry]->[X]--->[Measure]

    with device.using_qubit() as q:
        q:SimulatedQubit
        q.rotate_x(math.pi/4) # Apply Rx to qubit
        print('Apply Rx(pi/4)')
        print(f'P(|0>={q.ket0_p()}')
        print(f'P(|1>={q.ket1_p()}')
        print('Apply X')
        q.x()
        print(f'P(|0>={q.ket0_p()}')
        print(f'P(|1>={q.ket1_p()}')

if __name__ == '__main__':
    qsim = SingleQubitSimulator()
    apply_x_example(qsim)