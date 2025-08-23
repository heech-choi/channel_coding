
import numpy as np
from .base import BaseCode

class Uncoded(BaseCode):
    def name(self):
        return "Uncoded (rate 1)"

    def rate(self):
        return 1.0

    def encode(self, bits: np.ndarray) -> np.ndarray:
        return bits.astype(np.uint8)

    def decode(self, received: np.ndarray, **kwargs) -> np.ndarray:
        # For uncoded, received is already hard bits
        return received.astype(np.uint8)
