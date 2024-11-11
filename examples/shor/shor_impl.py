from typing import List

import numpy as np

from core.basic import X, HN, PAULI_X, KET_0, KET_1, H, CNOT, PAULI_Z, RX, CNOT_matr
from core.oracle import generate_oracle_simon
from core.simulator import NQubitSimulator

def shor(sim: NQubitSimulator):
    """
    Quantum Shor Algorithm
    Circuit: TODO...
    """
    pass

if __name__ == '__main__':
    sim = NQubitSimulator(3)
    shor(sim)






