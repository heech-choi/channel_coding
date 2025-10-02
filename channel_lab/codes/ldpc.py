
import numpy as np
from .base import BaseCode
# from utilities.log import parse_h_matrix

class LDPCCode(BaseCode):
    def __init__(self, n: int = 3):
        assert n >= 1 and n % 2 == 1, "n must be an odd integer >= 1"
        self.n = n
        # H_matrix = parse_h_matrix(input("H-matrix: "))
        print(bits)

    def name(self):
        return f"LDPC(n={self.n})"

    def rate(self):
        return 1.0 / self.n

    def encode(self, bits: np.ndarray) -> np.ndarray:
        return np.repeat(bits.astype(np.uint8), self.n) #np.unit8 uses 1byte - the cheapest. int takes 24+ bytes.

    def decode(self, received: np.ndarray, **kwargs) -> np.ndarray: #**kwargs is 'keyword arguments', where ** accepts any number of extra named arguments and bundle them into a dictionary.
        # majority vote over each n-block
        n = self.n
        assert received.size % n == 0, "length not divisible by n"
        blocks = received.reshape(-1, n)
        sums = np.sum(blocks, axis=1)
        hard = (sums >= (n//2 + 1)).astype(np.uint8) #sum-decoding here would yield different results than majority vote at huge outlier - e.g. -10, 2, 1 
        return hard

# def bit_flipping_decoding(bits:np.ndarray, h_matrix:np.ndarray):