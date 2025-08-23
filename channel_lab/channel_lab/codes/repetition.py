
import numpy as np
from .base import BaseCode

class RepetitionCode(BaseCode):
    def __init__(self, n: int = 3):
        assert n >= 1 and n % 2 == 1, "n must be an odd integer >= 1"
        self.n = n

    def name(self):
        return f"Repetition(n={self.n})"

    def rate(self):
        return 1.0 / self.n

    def encode(self, bits: np.ndarray) -> np.ndarray:
        return np.repeat(bits.astype(np.uint8), self.n)

    def decode(self, received: np.ndarray, **kwargs) -> np.ndarray:
        # majority vote over each n-block
        n = self.n
        assert received.size % n == 0, "length not divisible by n"
        blocks = received.reshape(-1, n)
        sums = np.sum(blocks, axis=1)
        hard = (sums >= (n//2 + 1)).astype(np.uint8)
        return hard
