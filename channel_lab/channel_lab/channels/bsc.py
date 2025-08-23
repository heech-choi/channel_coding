
import numpy as np

def transmit(bits: np.ndarray, p: float) -> np.ndarray:
    """
    Binary Symmetric Channel: flip each bit with probability p.
    """
    flips = (np.random.rand(bits.size) < p).astype(np.uint8)
    return (bits ^ flips).astype(np.uint8)
