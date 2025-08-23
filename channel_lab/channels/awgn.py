
import numpy as np

def bpsk_mod(bits: np.ndarray, ebn0_db: float, rate: float):
    """
    Map 0->+1, 1->-1, add AWGN with variance from Eb/N0 and code rate.
    SNR per bit: Eb/N0. For BPSK, symbol energy Es = Eb (m=1). 
    Noise variance per dimension: N0/2. If Eb/N0 = gamma, then sigma^2 = N0/2 = 1/(2 * gamma).
    We set signal amplitude A=1, so Eb = 1 (per information bit) and account for code rate in Es: Eb_coded = Eb / rate.
    => Effective SNR at symbol uses Eb_coded.
    """
    bits = bits.astype(np.uint8)
    x = 1 - 2*bits  # 0->+1, 1->-1
    gamma = 10**(ebn0_db/10.0)
    # Effective Eb/N0 per coded bit:
    gamma_eff = gamma * rate
    sigma2 = 1.0/(2.0*gamma_eff)
    noise = np.sqrt(sigma2) * np.random.randn(x.size)
    y = x + noise
    return y

def hard_demod(y: np.ndarray) -> np.ndarray:
    return (y < 0).astype(np.uint8)
