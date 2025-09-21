import numpy as np

# --------------- Get string input for matrix ------------

def get_input_for_matrix():
    print("Enter H-matrix like 101/010/111  (or 'q' to quit)")
    s = input("H-matrix: ").strip()
    if s.lower() in {"q", "quit", "exit"}:
        s = False
    return s

# ---------------- Input string to matrix ----------------

def convert_str_input_into_matrix(h_string: str) -> np.ndarray:
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

    return np.array(matrix, dtype=np.uint8)


# ----------------- print each step -----------------

def _print_step(title, H, row_ops=None, verbose=True):
    if not verbose:
        return
    print(f"\n=== {title} ===")
    if row_ops:
        print("Row op:", row_ops)
    print(H)



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

# def _print_step(title: str, H: np.ndarray, row_ops=None, col_ops=None, verbose=True):
#     if not verbose: 
#         return
#     print(f"\n=== {title} ===")
#     if row_ops:  print("Row op:", row_ops)
#     if col_ops:  print("Col op:", col_ops)
#     print(H)
