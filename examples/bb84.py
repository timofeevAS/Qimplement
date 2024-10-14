import random
from typing import List

from core.interface import QuantumDevice
from core.simulator import SimulatedQubit, SingleQubitSimulator

EVA_STORAGE: List[SimulatedQubit] = []
EVA_KEY = []
ALICE_KEY = []
BOB_KEY = []
ALICE_BASIS = []
BOB_BASIS = []

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

def eva_attack(q: SimulatedQubit, eva_device: QuantumDevice) -> SimulatedQubit:
    EVA_STORAGE.append(q.copy())
    [eva_message, eva_basis] = [
        sample_random_bit(eva_device) for _ in range(2)
    ]
    prepare_message_qubit(eva_message, eva_basis, q)
    return q


def send_single_bit_with_bb84(
        alice_device: QuantumDevice,
        bob_device: QuantumDevice,
        eva_device: QuantumDevice) -> tuple:
    [alice_message, alice_basis] = [
        sample_random_bit(alice_device) for _ in range(2)
    ]
    ALICE_BASIS.append(alice_basis)

    bob_basis = sample_random_bit(bob_device)
    BOB_BASIS.append(bob_basis)

    q_copy: SimulatedQubit
    q: SimulatedQubit
    with alice_device.using_qubit() as q:
        prepare_message_qubit(alice_message, alice_basis, q)
        q_copy = q.copy()

    q = eva_attack(q_copy, eva_device)

    bob_result = measure_message_qubit(bob_basis, q)

    return (alice_message, alice_basis), (bob_result, bob_basis)


def simulate_bb84(n_bits: int) -> list:
    alice_device = SingleQubitSimulator()
    bob_device = SingleQubitSimulator()
    eva_device = SingleQubitSimulator()

    key = []
    n_rounds = 0

    while len(key) < n_bits:
        n_rounds += 1
        ((alice_bit, alice_basis), (bob_result, bob_basis)) = \
            send_single_bit_with_bb84(alice_device, bob_device, eva_device)

        if alice_basis == bob_basis:
            # assert alice_bit == bob_result:
            EVA_KEY.append(measure_message_qubit(alice_basis,EVA_STORAGE[-1]))
            key.append(alice_bit)
            BOB_KEY.append(bob_result)

    print(f"Took {n_rounds} rounds to generate a {n_bits}-bit key.")

    return key


def apply_one_time_pad(message: List[bool], key: List[bool]) -> List[bool]:
    return [
        message_bit ^ key_bit
        for (message_bit, key_bit) in zip(message, key)
    ]


if __name__ == "__main__":
    print("Generating a 100-bit key by simulating BB84...")
    SIZE=100
    message = list(map(bool, [random.randint(0, 1) for _ in range(SIZE)]))
    key = simulate_bb84(SIZE)
    print(f"Eva key                           {convert_to_hex(EVA_KEY)}.")
    print(f"Bob key                           {convert_to_hex(BOB_KEY)}.")
    print(f"Alice key                         {convert_to_hex(key)}.")
    print(f"Using key to send secret message: {convert_to_hex(message)}.")

    encrypted_message = apply_one_time_pad(message, key)
    print(f"Encrypted message:                {convert_to_hex(encrypted_message)}.")

    decrypted_message_bob = apply_one_time_pad(encrypted_message, BOB_KEY)
    decrypted_message_eva = apply_one_time_pad(encrypted_message, EVA_KEY)
    print(f"Bob decrypted to get:             {convert_to_hex(decrypted_message_bob)}.")
    print(f"Eva decrypted to get:             {convert_to_hex(decrypted_message_eva)}.")