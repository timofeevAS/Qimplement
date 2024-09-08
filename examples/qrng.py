from core.interface import QuantumDevice
from core.simulator import SingleQubitSimulator


def qrng(device: QuantumDevice) -> bool:
    with device.using_qubit() as q:
        q.h()
        return q.measure()


if __name__ == '__main__':
    qsim = SingleQubitSimulator()
    for i in range(10):
        sample = qrng(qsim)
        print(f'[x] QRNG value is: {int(sample)}')