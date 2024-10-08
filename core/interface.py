from abc import ABCMeta, abstractmethod
from contextlib import contextmanager

# TODO: create class for implement state of qubit
class QubitInterface(metaclass=ABCMeta):
    @abstractmethod
    def h(self): pass

    @abstractmethod
    def x(self): pass

    @abstractmethod
    def measure(self) -> bool: pass

    @abstractmethod
    def reset(self): pass

    @abstractmethod
    def pauli_x(self): pass

    @abstractmethod
    def pauli_y(self): pass

    @abstractmethod
    def pauli_z(self): pass

    @abstractmethod
    def rotate_x(self, angle): pass

    @abstractmethod
    def rotate_y(self, angle): pass

    @abstractmethod
    def rotate_z(self, angle): pass

    def ket0_p(self) -> float: pass

    def ket1_p(self) -> float: pass


class QuantumDevice(metaclass=ABCMeta):
    @abstractmethod
    def allocate_qubit(self) -> QubitInterface:
        pass

    @abstractmethod
    def deallocate_qubit(self, qubit: QubitInterface):
        pass

    @contextmanager
    def using_qubit(self) -> QubitInterface:
        qubit = self.allocate_qubit()
        try:
            yield qubit
        finally:
            qubit.reset()
            self.deallocate_qubit(qubit)
