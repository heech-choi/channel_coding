import numpy as np

def parse_h_matrix(h_string: str) -> np.ndarray:
    rows = h_string.strip().split("/")
    if not rows or any(r.strip() == "" for r in rows):
        raise ValueError("Input must contain one or more rows separated by '/'.")

    matrix = []
    for row in rows:
        clean = row.replace(" ", "").replace(",", "")
        if not clean or any(ch not in "01" for ch in clean):
            raise ValueError("Rows must contain only 0/1 digits (optionally spaced or comma-separated).")
        matrix.append([int(ch) for ch in clean])

    # Check rectangular shape
    n_cols = len(matrix[0])
    if not all(len(r) == n_cols for r in matrix):
        raise ValueError("All rows must have the same length.")

    return np.array(matrix, dtype=np.unit8)

# #REPL-style loop
# print("Enter H-matrix like 101/010/111  (or 'q' to quit)")
# while True:
#     try:
#         s = input("H-matrix: ").strip()
#         if s.lower() in {"q", "quit", "exit"}:
#             print("Bye!")
#             break

#         H = parse_h_matrix(s)         
#         print(H)                       

#     except ValueError as e:
#         print(f"[Input error] {e}")
#         continue
#     except KeyboardInterrupt:
#         print("\nInterrupted. Bye!")
#         break
#     except EOFError:
#         print("\nEOF. Bye!")
#         break

def _print_step(title: str, H: np.ndarray, row_ops=None, col_ops=None, verbose=True):
    if not verbose: 
        return
    print(f"\n=== {title} ===")
    if row_ops:  print("Row op:", row_ops)
    if col_ops:  print("Col op:", col_ops)
    print(H)

def h_to_standard_form(H: np.ndarray, verbose: bool = True):
    """
    Reduce a binary parity-check matrix H over GF(2) to strict standard form [A | I],
    printing each row/column operation. It locks pivot columns as it proceeds so the
    right block becomes an actual identity (not just a permutation).
    
    Returns:
        H_std : matrix in [A | I] form
        col_perm : permutation such that H[:, col_perm] == H_std
    """
    H = (H.copy() % 2).astype(np.uint8)
    m, n = H.shape
    if m > n:
        raise ValueError("Need m <= n.")

    col_perm = np.arange(n, dtype=int)

    def swap_rows(i, j):
        if i == j: return
        H[[i, j]] = H[[j, i]]
        _print_step(f"swap rows r{i}<->r{j}", H, row_ops=f"r{i}<->r{j}", verbose=verbose)

    def swap_cols(i, j):
        if i == j: return
        H[:, [i, j]] = H[:, [j, i]]
        col_perm[[i, j]] = col_perm[[j, i]]
        _print_step(f"swap cols c{i}<->c{j}", H, col_ops=f"c{i}<->c{j}", verbose=verbose)

    right_start = n - m
    # For i = 0..m-1, make column (right_start + i) the i-th identity column
    for i in range(m):
        target_col = right_start + i

        # 1) Find a pivot column among the remaining right-block columns [target_col .. n)
        pivot_col = None
        for j in range(target_col, n):
            if H[i:, j].any():
                pivot_col = j
                break
        if pivot_col is None:
            raise ValueError("Cannot form [A|I]: rank deficiency in right block.")

        # 2) Move that pivot column into target_col (does not touch earlier pivots)
        if pivot_col != target_col:
            swap_cols(pivot_col, target_col)

        # 3) Ensure pivot 1 at (i, target_col) by row swap if needed
        if H[i, target_col] == 0:
            for r in range(i + 1, m):
                if H[r, target_col] == 1:
                    swap_rows(i, r)
                    break
        if H[i, target_col] == 0:
            # Should not happen given the any() check, but guard anyway
            raise ValueError("Internal error: couldn't position pivot 1.")

        # 4) Eliminate other 1s in target_col
        for r in range(m):
            if r != i and H[r, target_col]:
                H[r] ^= H[i]
                _print_step(f"r{r} ^= r{i}", H, row_ops=f"r{r} ^= r{i}", verbose=verbose)

    # Sanity check
    right = H[:, right_start:]
    if not np.array_equal(right, np.eye(m, dtype=np.uint8)):
        raise ValueError("Unexpected: right block is not identity.")

    _print_step("Final [A | I]", H, verbose=verbose)
    return H, col_perm


def generator_from_parity(H_std: np.ndarray) -> np.ndarray:
    """
    Given H in standard form [A | I], return G = [I | A^T].
    """
    H_std = H_std % 2
    m, n = H_std.shape
    k = n - m
    A = H_std[:, :k]
    G = np.concatenate((np.eye(k, dtype=np.uint8), (A.T % 2)), axis=1)
    return G % 2

if __name__ == "__main__":
    H = np.array([
        [1,0,1,1,0,0],
        [0,1,1,0,1,0],
        [1,1,0,0,0,1],
    ], dtype=np.uint8)

    H_std, perm = h_to_standard_form(H, verbose=True)
    print("\nColumn permutation (original -> new position):")
    print(perm)

    G = generator_from_parity(H_std)
    print("\nGenerator matrix G = [I | A^T]:")
    print(G)