from core.interface import QuantumDevice
from core.simulator import SingleQubitSimulator


def qrng(device: QuantumDevice) -> bool:
    with device.using_qubit() as q:
        q.h() # Apply Hadamard matrix to qubit
        return q.measure() # Measure qubit to get zero or one (True or False)


if __name__ == '__main__':
    qsim = SingleQubitSimulator()
    for i in range(10):
        sample = qrng(qsim)
        print(f'[x] QRNG value is: {int(sample)}')