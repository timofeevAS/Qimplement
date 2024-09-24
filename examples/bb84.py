import random
from typing import List

from core.interface import QuantumDevice
from core.simulator import SimulatedQubit, SingleQubitSimulator


def sample_random_bit(device: QuantumDevice) -> bool:
    with device.using_qubit() as q:
        q.h()
        result = q.measure()
        q.reset()
    return result


def prepare_message_qubit(message: bool, basis: bool, q: SimulatedQubit) -> None:
    if message:
        q.x()
    if basis:
        q.h()


def measure_message_qubit(basis: bool, q: SimulatedQubit) -> bool:
    if basis:
        q.h()
    result = q.measure()
    q.reset()
    return result


def convert_to_hex(bits: List[bool]) -> str:
    return hex(int(
        "".join(["1" if bit else "0" for bit in bits]),
        2
    ))


def send_single_bit_with_bb84(
        alice_device: QuantumDevice,
        bob_device: QuantumDevice
) -> tuple:
    [your_message, your_basis] = [
        sample_random_bit(alice_device) for _ in range(2)
    ]

    bob_basis = sample_random_bit(bob_device)

    with alice_device.using_qubit() as q:
        prepare_message_qubit(your_message, your_basis, q)

        # QUBIT SENDING...

        bob_result = measure_message_qubit(bob_basis, q)

    return (your_message, your_basis), (bob_result, bob_basis)


def simulate_bb84(n_bits: int) -> list:
    alice_device = SingleQubitSimulator()
    bob_device = SingleQubitSimulator()

    key = []
    n_rounds = 0

    while len(key) < n_bits:
        n_rounds += 1
        ((your_bit, your_basis), (eve_result, eve_basis)) = \
            send_single_bit_with_bb84(alice_device, bob_device)

        if your_basis == eve_basis:
            assert your_bit == eve_result
            key.append(your_bit)

    print(f"Took {n_rounds} rounds to generate a {n_bits}-bit key.")

    return key


def apply_one_time_pad(message: List[bool], key: List[bool]) -> List[bool]:
    return [
        message_bit ^ key_bit
        for (message_bit, key_bit) in zip(message, key)
    ]


if __name__ == "__main__":
    print("Generating a 100-bit key by simulating BB84...")
    message = list(map(bool, [random.randint(0, 1) for _ in range(100)]))
    key = simulate_bb84(100)
    print(f"Got key                           {convert_to_hex(key)}.")
    print(f"Using key to send secret message: {convert_to_hex(message)}.")

    encrypted_message = apply_one_time_pad(message, key)
    print(f"Encrypted message:                {convert_to_hex(encrypted_message)}.")

    decrypted_message = apply_one_time_pad(encrypted_message, key)
    print(f"Bob decrypted to get:             {convert_to_hex(decrypted_message)}.")

    assert convert_to_hex(message) == convert_to_hex(decrypted_message)