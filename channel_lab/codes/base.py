
from abc import ABC, abstractmethod
import numpy as np

class BaseCode(ABC):
    """
    Abstract base class for channel codes.
    Implement encode() and decode() on bit-level numpy arrays of shape (N,).
    """
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def rate(self) -> float:
        ...

    @abstractmethod
    def encode(self, bits: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def decode(self, received: np.ndarray, **kwargs) -> np.ndarray:
        """Return hard-decoded information bits as ndarray of 0/1"""
        ...
