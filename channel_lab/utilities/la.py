import numpy as np

import numpy as np

# -----------------RREF ----------------------

def rref_mod2(H):
    """
    Row-Reduced Echelon Form over GF(2).
    Returns:
      R: RREF(H) over GF(2)
      pivot_cols: list of pivot column indices
    """
    H = (np.array(H, dtype=np.uint8) & 1).copy()
    m, n = H.shape
    R = H.copy()
    pivot_cols = []
    row = 0
    for col in range(n):
        # find a row with a 1 in (row..m-1, col)
        pivot = None
        for r in range(row, m):
            if R[r, col]:
                pivot = r
                break
        if pivot is None:
            continue  # no pivot in this column

        # swap pivot row into current 'row'
        if pivot != row:
            R[[row, pivot]] = R[[pivot, row]]

        # clear all other 1s in this column
        for r in range(m):
            if r != row and R[r, col]:
                R[r, :] ^= R[row, :]  # add (XOR) pivot row

        pivot_cols.append(col)
        row += 1
        if row == m:
            break

    return R, pivot_cols

# ---------------- RREF -> Nullspace basis ------------

def nullspace_basis_from_rref(R, pivot_cols, n):
    """
    Given RREF matrix R over GF(2) and pivot columns,
    construct a basis for Null(H) (rows of G).
    Works even if rank < #rows.
    """

    print("RREFed H =\n", R)
    pivot_set = set(pivot_cols)
    free_cols = [j for j in range(n) if j not in pivot_set]
    m = R.shape[0]

    basis = []
    # For each free variable f, build one solution vector v
    for f in free_cols:
        v = np.zeros(n, dtype=np.uint8)
        v[f] = 1
        # For each pivot row i with pivot in pc, set v[pc] = R[i, f]
        for i, pc in enumerate(pivot_cols):
            v[pc] = R[i, f] & 1
        basis.append(v)
    if not basis:
        # Nullspace is {0}; return a (0 x n) matrix
        G = np.zeros((0, n), dtype=np.uint8)
    else:
        G = np.vstack(basis)
    return G, free_cols


# ------------- Generic G: RREF, nullspace, Basis -------------------

def generator_from_H(H):
    """
    General method: compute a generator matrix G (rows form a basis of Null(H))
    for ANY binary parity-check matrix H over GF(2).
    Returns:
      G  : (k x n) generator
      piv: pivot column indices
      free: free column indices
    """
    H = (np.array(H, dtype=np.uint8) & 1)
    m, n = H.shape
    R, piv = rref_mod2(H)
    G, free = nullspace_basis_from_rref(R, piv, n)
    return G, piv, free

# --------------- Systematic G : [I|A] by column permutation  -------------

def try_systematic_G(H):
    """
    Try to produce a systematic generator matrix.

    Steps:
    - RREF(H) to find pivot columns (rank = r).
    - Permute columns so pivot columns come first: H_perm = [I_r | A] in RREF.
    - Then G_sys (in permuted coordinates) = [A^T | I_k], where k = n - r.
    - Return both the permuted G (systematic order) and the unpermuted G (original order).

    Returns:
      G_sys_perm   : (k x n) in systematic column order (pivot-first)
      G_sys_unperm : (k x n) mapped back to original column order
      perm         : list of column indices (new_order[j] = old_index)
      invperm      : inverse permutation (old_index -> position in permuted order)
      piv, free    : pivot and free column indices (w.r.t. original H)
    """
    H = (np.array(H, dtype=np.uint8) & 1)
    m, n = H.shape
    R, piv = rref_mod2(H)
    r = len(piv)
    k = n - r
    free = [j for j in range(n) if j not in set(piv)]

    # Permutation: pivot columns first, then free columns
    perm = piv + free
    invperm = np.zeros(n, dtype=int)
    for new_pos, old_idx in enumerate(perm):
        invperm[old_idx] = new_pos

    # Permute columns of RREF(H) to get [I_r | A]
    R_perm = R[:, perm]
    # In exact RREF, the left block should be I_r (for the pivot rows)
    # Extract A: shape r x k
    A = R_perm[:, r:]

    # Systematic G in permuted coords: [A^T | I_k]
    G_sys_perm = np.hstack([A.T & 1, np.eye(k, dtype=np.uint8)])

    # Map G back to original column order:
    # Columns of G_sys_perm are in 'perm' order; we need to place them back.
    G_sys_unperm = np.zeros_like(G_sys_perm)
    for j_old in range(n):
        j_new = invperm[j_old]
        G_sys_unperm[:, j_old] = G_sys_perm[:, j_new]

    return G_sys_perm, G_sys_unperm, perm, invperm, piv, free

# ----------- check HG^T = 0 --------------------------

def check_HG_zero(H, G):
    """Return True if H G^T == 0 over GF(2)."""
    H = (np.array(H, dtype=np.uint8) & 1)
    G = (np.array(G, dtype=np.uint8) & 1)
    return np.all((H @ G.T) % 2 == 0)


# ------------get Rank of gf(2) matrix ----------------

def gf2_rank(A: np.ndarray) -> int:
    """Rank over GF(2) via elimination (destroys A copy)."""
    A = (A.copy() % 2).astype(np.uint8)
    m, n = A.shape
    r = 0
    for c in range(n):
        # find pivot row
        pivot = None
        for i in range(r, m):
            if A[i, c] == 1:
                pivot = i
                break
        if pivot is None:
            continue
        # swap to row r
        if pivot != r:
            A[[r, pivot]] = A[[pivot, r]]
        # eliminate below
        for i in range(r + 1, m):
            if A[i, c] == 1:
                A[i] ^= A[r]
        r += 1
        if r == m:
            break
    return r

# ------------------invertibility check for right block (for systematic form) ------------

def right_block_invertible(H: np.ndarray) -> bool:
    m, n = H.shape
    B = H[:, n - m :]
    return gf2_rank(B) == m

# ----------------- Algorithm to create [A | I]-----------------

def to_standard_form_rows_only(H: np.ndarray, verbose: bool = True):
    """
    Convert binary H to [A | I] using ONLY row operations.
    Preconditions: m <= n and the right m x m block is invertible over GF(2).

    Returns:
        H_std: row-equivalent matrix with right block = I
    """
    H = (H.copy() % 2).astype(np.uint8)
    m, n = H.shape
    if m > n:
        raise ValueError("Require m <= n.")

    right_start = n - m

    # We'll Gauss-Jordan the right block to identity using row ops,
    # applying those same ops to the ENTIRE H.
    for i in range(m):
        # 1) Ensure a pivot 1 at (i, right_start + i) using row swaps (if needed)
        if H[i, right_start + i] == 0:
            pivot_row = None
            for r in range(i + 1, m):
                if H[r, right_start + i] == 1:
                    pivot_row = r
                    break
            if pivot_row is None:
                # If this happens, right block isn't invertible under rows-only ops
                raise ValueError("Right block is singular; cannot form [A|I] using rows only.")
            # swap rows i <-> pivot_row
            H[[i, pivot_row]] = H[[pivot_row, i]]
            _print_step(f"swap r{i} <-> r{pivot_row}", H, row_ops=f"r{i}<->r{pivot_row}", verbose=verbose)

        # 2) Clear all other 1s in this pivot column via XOR row adds
        for r in range(m):
            if r != i and H[r, right_start + i] == 1:
                H[r] ^= H[i]
                _print_step(f"r{r} ^= r{i}", H, row_ops=f"r{r} ^= r{i}", verbose=verbose)

    # Sanity check: right block is identity
    right = H[:, right_start:]
    if not np.array_equal(right, np.eye(m, dtype=np.uint8)):
        raise RuntimeError("Unexpected failure: right block is not identity at the end.")

    _print_step("Final [A | I]", H, verbose=verbose)
    return H

# ----------------- Generator Matrix -----------------

def generator_from_parity(H_std: np.ndarray) -> np.ndarray:
    """
    Given H in standard form [A | I], return G = [I | A^T] over GF(2).
    """
    H_std = (H_std % 2).astype(np.uint8)
    m, n = H_std.shape
    k = n - m
    A = H_std[:, :k]
    G = np.concatenate([np.eye(k, dtype=np.uint8), (A.T % 2)], axis=1)
    return G % 2

# ----------------- Random H with invertible right block -----------------

def random_H_with_invertible_right(m: int, n: int, seed: int | None = None) -> np.ndarray:
    """
    Sample random H ~ Bernoulli(0.5) until the right m x m block is invertible (GF(2)).
    No column permutations are used; we simply resample H if needed.
    """
    if m > n:
        raise ValueError("Need m <= n.")
    rng = np.random.default_rng(seed)
    while True:
        H = rng.integers(0, 2, size=(m, n), dtype=np.uint8)
        if right_block_invertible(H):
            return H


# def h_to_standard_form(H: np.ndarray, verbose: bool = True):
#     """
#     Reduce a binary parity-check matrix H over GF(2) to strict standard form [A | I],
#     printing each row/column operation. It locks pivot columns as it proceeds so the
#     right block becomes an actual identity (not just a permutation).
    
#     Returns:
#         H_std : matrix in [A | I] form
#         col_perm : permutation such that H[:, col_perm] == H_std
#     """
#     H = (H.copy() % 2).astype(np.uint8)
#     m, n = H.shape
#     if m > n:
#         raise ValueError("Need m <= n.")

#     col_perm = np.arange(n, dtype=int)

#     def swap_rows(i, j):
#         if i == j: return
#         H[[i, j]] = H[[j, i]]
#         _print_step(f"swap rows r{i}<->r{j}", H, row_ops=f"r{i}<->r{j}", verbose=verbose)

#     def swap_cols(i, j):
#         if i == j: return
#         H[:, [i, j]] = H[:, [j, i]]
#         col_perm[[i, j]] = col_perm[[j, i]]
#         _print_step(f"swap cols c{i}<->c{j}", H, col_ops=f"c{i}<->c{j}", verbose=verbose)

#     right_start = n - m
#     # For i = 0..m-1, make column (right_start + i) the i-th identity column
#     for i in range(m):
#         target_col = right_start + i

#         # 1) Find a pivot column among the remaining right-block columns [target_col .. n)
#         pivot_col = None
#         for j in range(target_col, n):
#             if H[i:, j].any():
#                 pivot_col = j
#                 break
#         if pivot_col is None:
#             raise ValueError("Cannot form [A|I]: rank deficiency in right block.")

#         # 2) Move that pivot column into target_col (does not touch earlier pivots)
#         if pivot_col != target_col:
#             swap_cols(pivot_col, target_col)

#         # 3) Ensure pivot 1 at (i, target_col) by row swap if needed
#         if H[i, target_col] == 0:
#             for r in range(i + 1, m):
#                 if H[r, target_col] == 1:
#                     swap_rows(i, r)
#                     break
#         if H[i, target_col] == 0:
#             # Should not happen given the any() check, but guard anyway
#             raise ValueError("Internal error: couldn't position pivot 1.")

#         # 4) Eliminate other 1s in target_col
#         for r in range(m):
#             if r != i and H[r, target_col]:
#                 H[r] ^= H[i]
#                 _print_step(f"r{r} ^= r{i}", H, row_ops=f"r{r} ^= r{i}", verbose=verbose)

#     # Sanity check
#     right = H[:, right_start:]
#     if not np.array_equal(right, np.eye(m, dtype=np.uint8)):
#         raise ValueError("Unexpected: right block is not identity.")

#     _print_step("Final [A | I]", H, verbose=verbose)
#     return H, col_perm


# def generator_from_parity(H_std: np.ndarray) -> np.ndarray:
#     """
#     Given H in standard form [A | I], return G = [I | A^T].
#     """
#     H_std = H_std % 2
#     m, n = H_std.shape
#     k = n - m
#     A = H_std[:, :k]
#     G = np.concatenate((np.eye(k, dtype=np.uint8), (A.T % 2)), axis=1)
#     return G % 2

# if __name__ == "__main__":
#     H = np.array([
#         [1,0,1,1,0,0],
#         [0,1,1,0,1,0],
#         [1,1,0,0,0,1],
#     ], dtype=np.uint8)

#     H_std, perm = h_to_standard_form(H, verbose=True)
#     print("\nColumn permutation (original -> new position):")
#     print(perm)

#     G = generator_from_parity(H_std)
#     print("\nGenerator matrix G = [I | A^T]:")
#     print(G)