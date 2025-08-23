
import numpy as np
from typing import Callable, Dict, Any, Tuple

def ber_curve_awgn(code, ebn0_db_list, n_bits=10000, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate BER vs Eb/N0 for BI-AWGN + BPSK + hard decision.
    code: object with encode(bits)->coded_bits and decode(received_bits)->decoded_bits
    """
    if seed is not None:
        np.random.seed(seed)
    from channels.awgn import bpsk_mod, hard_demod

    ber = []
    ebn0_db_arr = np.array(list(ebn0_db_list), dtype=float)
    for ebn0_db in ebn0_db_arr:
        # generate information bits
        u = np.random.randint(0, 2, size=n_bits, dtype=np.uint8)
        c = code.encode(u)
        y = bpsk_mod(c, ebn0_db=ebn0_db, rate=code.rate())
        r_hard = hard_demod(y)
        u_hat = code.decode(r_hard)
        errors = np.count_nonzero(u_hat ^ u)
        ber.append(errors / n_bits)
    return ebn0_db_arr, np.array(ber, dtype=float)

def ber_curve_bsc(code, p_list, n_bits=10000, seed: int = None):
    if seed is not None:
        np.random.seed(seed)
    from channels.bsc import transmit

    ber = []
    p_arr = np.array(list(p_list), dtype=float)
    for p in p_arr:
        u = np.random.randint(0, 2, size=n_bits, dtype=np.uint8)
        c = code.encode(u)
        r = transmit(c, p=p)
        u_hat = code.decode(r)
        errors = np.count_nonzero(u_hat ^ u)
        ber.append(errors / n_bits)
    return p_arr, np.array(ber, dtype=float)
