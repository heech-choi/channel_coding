
import numpy as np

def bpsk_mod(bits: np.ndarray, ebn0_db: float, rate: float):
    bits = bits.astype(np.uint8)
    x = 1 - 2*bits  # 0->+1, 1->-1
    gamma = 10**(ebn0_db/10.0)
    # Effective Eb/N0 per coded bit:
    gamma_eff = gamma * rate
    sigma2 = 1.0/(2.0*gamma_eff)
    noise = np.sqrt(sigma2) * np.random.randn(x.size)
    y = x + noise
    return y

def hard_demod(y: np.ndarray) -> np.ndarray: # -> is a kind notice that the result will have that type. small comment, not affecting runtime performance
    return (y < 0).astype(np.uint8)
