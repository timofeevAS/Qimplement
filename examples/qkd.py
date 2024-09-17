from core.interface import QuantumDevice
from core.simulator import SimulatedQubit, SingleQubitSimulator


def prepare_classical_message(bit: bool, q: SimulatedQubit) -> None:
    if bit:
        q.x()

def eve_measure(q: SimulatedQubit) -> bool:
    return q.measure()

def send_classical_bit(device: QuantumDevice, bit: bool) -> None:
    with device.using_qubit() as q:
        prepare_classical_message(bit, q)
        result = eve_measure(q)
        q.reset()
    assert result == bit

def prepare_classical_message_plusminus(bit: bool, q: SimulatedQubit) -> None:
    if bit:
        q.x()
    q.h()

def eve_measure_plusminus(q: SimulatedQubit) -> bool:
    q.h()
    return q.measure()

def send_classical_bit_plusminus(device: QuantumDevice, bit: bool) -> None:
    with device.using_qubit() as q:
        prepare_classical_message_plusminus(bit, q)
        result = eve_measure_plusminus(q)
        print(result)
        assert result == bit

def send_classical_bit_wrong_basis(device: QuantumDevice, bit: bool) -> None:
    with device.using_qubit() as q:
        prepare_classical_message(bit, q)
        result = eve_measure_plusminus(q)
        print(result)
        assert result == bit, "Two parties do not have the same bit value"

if __name__ == '__main__':
    qsim = SingleQubitSimulator()
    send_classical_bit_plusminus(qsim, False)
    send_classical_bit_plusminus(qsim, True)

    send_classical_bit_wrong_basis(qsim, False) # assertion
    send_classical_bit_wrong_basis(qsim, True) # assertion