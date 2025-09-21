import numpy as np
from typing import Iterable, Tuple, List, Union

def _coerce_y(y_in: Union[str, Iterable, np.ndarray]) -> np.ndarray:
    """
    Coerce y into a 1D numpy array of {0,1}.
    - If string like "101011", parse digits.
    - If floats (e.g., BPSK or soft), hard-decision at 0: y_i=0 if val>=0 else 1.
    - If {-1,+1} values, same rule as floats.
    """
    if isinstance(y_in, str):
        y = np.array([int(ch) for ch in y_in.strip() if ch in "01"], dtype=int)
        if y.size == 0:
            raise ValueError("String y must contain digits 0/1.")
        return y

    y = np.asarray(y_in).reshape(-1)
    # If already in {0,1}, return
    if np.all((y == 0) | (y == 1)):
        return y.astype(int)

    # Otherwise treat as real-valued BPSK-like hard decision
    y_hd = (y < 0).astype(int)  # bit=0 if value>=0, else 1
    return y_hd

def _build_neighbors(H: np.ndarray):
    """Precompute neighbor lists for checks and variables."""
    m, n = H.shape
    check_neighbors = [np.where(H[i] == 1)[0] for i in range(m)]
    var_neighbors   = [np.where(H[:, j] == 1)[0] for j in range(n)]
    return check_neighbors, var_neighbors

def _print_check_details(iter_idx: int, H: np.ndarray, y: np.ndarray,
                         check_neighbors: List[np.ndarray], syndrome: np.ndarray):
    print(f"\n[Iteration {iter_idx}] Check-node evaluation:")
    for i, N_i in enumerate(check_neighbors):
        vals = y[N_i]
        parity = int(vals.sum() % 2)
        status = "UNSAT ❌" if parity == 1 else "SAT ✅"
        print(f"  C{i}: indices {list(N_i)} | bits {list(vals)} | sum%2={parity} -> {status}")

def _print_variable_details(iter_idx: int, y: np.ndarray, bit_scores: np.ndarray,
                            var_neighbors: List[np.ndarray], syndrome: np.ndarray,
                            flip_set: np.ndarray, rule: str):
    print(f"\n[Iteration {iter_idx}] Variable-node scoring & decisions (rule='{rule}'):")
    for j, N_j in enumerate(var_neighbors):
        unsat_checks = [int(i) for i in N_j if syndrome[i] == 1]
        deg = len(N_j)
        print(f"  V{j}: bit={y[j]} | checks {list(N_j)} | UNSAT checks {unsat_checks} "
              f"| score={bit_scores[j]} | deg={deg} | flip={'YES' if j in flip_set else 'no'}")

def bit_flipping_decode(
    H: np.ndarray,
    y_in: Union[str, Iterable, np.ndarray],
    max_iters: int = 20,
    rule: str = "max",            # 'max' or 'threshold'
    flip_all_max: bool = True,    # only used if rule == 'max'
    prevent_oscillation: bool = False,  # if True, avoid flipping a bit in two consecutive iters
    verbose: bool = True
) -> Tuple[np.ndarray, bool, int]:
    """
    Bit-flipping decoder with detailed per-node logging.

    Args:
        H: (m x n) parity-check matrix over GF(2) with entries 0/1.
        y_in: received word (0/1 list/array, a '101011' string, or real/BPSK soft values).
        max_iters: maximum iterations.
        rule:
            - 'max': flip all bits whose score equals the maximum score (>0).
            - 'threshold': flip bits with score > degree/2 (classic Gallager bit-flip).
        flip_all_max: when rule='max', flip all ties if True; otherwise flip only the lowest index.
        prevent_oscillation: if True, avoid flipping a bit in two consecutive iterations.
        verbose: print detailed logs.

    Returns:
        (decoded_word, success_flag, iterations_used)
    """
    H = np.asarray(H, dtype=int)
    if H.ndim != 2:
        raise ValueError("H must be a 2D array.")
    if not np.all((H == 0) | (H == 1)):
        raise ValueError("H must contain only 0/1.")

    y = _coerce_y(y_in).astype(int).copy()
    m, n = H.shape
    if y.size != n:
        raise ValueError(f"Length of y ({y.size}) must match number of columns in H ({n}).")

    check_neighbors, var_neighbors = _build_neighbors(H)
    last_flipped = np.zeros(n, dtype=bool)

    if verbose:
        print("=== Bit-Flipping Decoder (detailed) ===")
        print(f"H shape: {H.shape}, y length: {len(y)}")
        print(f"Initial y: {list(y)}")
        print(f"Rule: '{rule}' | flip_all_max={flip_all_max} | prevent_oscillation={prevent_oscillation}")

    for it in range(1, max_iters + 1):
        # Syndrome
        syndrome = (H @ y) % 2

        if verbose:
            print(f"\n=== Iteration {it} ===")
            print(f"Current y: {list(y)}")
            print(f"Syndrome: {list(syndrome)}")

        if np.all(syndrome == 0):
            if verbose:
                print("All parity checks satisfied. ✅ Decoding success.")
            return y, True, it - 1

        # Check-node details
        if verbose:
            _print_check_details(it, H, y, check_neighbors, syndrome)

        # Variable-node scores
        bit_scores = np.zeros(n, dtype=int)
        degrees = np.zeros(n, dtype=int)
        for j in range(n):
            N_j = var_neighbors[j]
            degrees[j] = len(N_j)
            if degrees[j] > 0:
                bit_scores[j] = int(np.sum(syndrome[N_j]))
            else:
                bit_scores[j] = 0

        # Decide flip set
        flip_set: np.ndarray
        if rule == "threshold":
            # Flip if more than half of connected checks are UNSAT
            # deg 0 -> never flips
            flip_mask = np.array([bit_scores[j] > (degrees[j] / 2.0) if degrees[j] > 0 else False
                                  for j in range(n)])
            flip_set = np.where(flip_mask)[0]
        elif rule == "max":
            max_score = int(bit_scores.max())
            if max_score <= 0:
                if verbose:
                    print("No bit has a positive score. Stopping (no progress possible).")
                return y, False, it
            idx = np.where(bit_scores == max_score)[0]
            if not flip_all_max:
                idx = np.array([idx.min()])  # deterministic tie-break: pick lowest index
            flip_set = idx
        else:
            raise ValueError("rule must be 'max' or 'threshold'")

        # Optional oscillation guard
        if prevent_oscillation and it > 1:
            flip_set = np.array([j for j in flip_set if not last_flipped[j]])

        # Variable-node details + flip decision summary
        if verbose:
            _print_variable_details(it, y, bit_scores, var_neighbors, syndrome, flip_set, rule)
            print(f"\n[Iteration {it}] Bits selected to flip: {list(flip_set)}")

        if flip_set.size == 0:
            if verbose:
                print("Empty flip set. Stopping (stuck).")
            return y, False, it

        # Apply flips
        y[flip_set] ^= 1

        # Track which bits flipped this round
        this_flipped = np.zeros(n, dtype=bool)
        this_flipped[flip_set] = True
        last_flipped = this_flipped

        if verbose:
            print(f"[Iteration {it}] y after flipping: {list(y)}")

    if verbose:
        print("\nReached max_iters without satisfying all checks.")
    return y, False, max_iters


# ------------------------------- Demo -------------------------------
if __name__ == "__main__":
    # Example H (3x6). Replace with your own.
    # H = np.array([
    #     [1, 0, 1, 0, 0, 1],
    #     [0, 1, 1, 0, 1, 0],
    #     [0, 0, 0, 1, 1, 1],
    # ], dtype=int)
    H = np.array([
        [1, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 1, 0],
        [1, 0, 0, 0, 1, 1],
        [0, 0, 1, 1, 0, 1],
    ], dtype=int)

    # Example received word (can be "101011" string, list, array, or soft values)
    y = "011011"

    decoded, success, iters = bit_flipping_decode(
        H, y,
        max_iters=20,
        rule="max",             # try "threshold" as well
        flip_all_max=True,      # flip all ties for 'max' rule
        prevent_oscillation=False,
        verbose=True            # set False to silence detailed logs
    )

    print("\nFinal decoded:", decoded.tolist())
    print("Decoding successful?", success)
    print("Iterations used:", iters)
