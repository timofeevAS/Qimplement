import math

from core.interface import QuantumDevice
from core.simulator import SingleQubitSimulator, SimulatedQubit


def qrng(device: QuantumDevice) -> bool:
    # Quantum circuit:
    # |0> --->[H]--->[Measure]

    with device.using_qubit() as q:
        q.h() # Apply Hadamard matrix to qubit
        return q.measure() # Measure qubit to get zero or one (True or False)


if __name__ == '__main__':
    qsim = SingleQubitSimulator()
    zero_count = 0
    one_count = 0

    total_count=1000
    for i in range(total_count):
        sample = qrng(qsim)
        if sample:
            one_count+=1
        else:
            zero_count+=1
        print(f'[x] QRNG value is: {int(sample)}')

    print(f'P(|1>) = {one_count/total_count}')