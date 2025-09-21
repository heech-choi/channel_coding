from log import * 
from la import *


# # A random-ish H (3 x 6)
# H = np.array([
#     [1,1,0,1,0,0],
#     [0,1,1,0,1,0],
#     [1,0,1,0,0,1],
# ], dtype=np.uint8)


H = get_input_for_matrix()
H = convert_str_input_into_matrix(H)
print("H =\n", H)

# General nullspace method
G, piv, free = generator_from_H(H)
print("\n[General] pivots:", piv, " | free:", free)
print("G (rows span Null(H)) =\n", G)
print("HG^T == 0 ?", check_HG_zero(H, G))

# # Try systematic form
# Gp, Gu, perm, invperm, piv2, free2 = try_systematic_G(H)
# print("\n[Systematic attempt] pivots:", piv2, " | free:", free2)
# print("Column order (pivot-first):", perm)
# print("G_sys (permuted) =\n", Gp)
# print("G_sys (original order) =\n", Gu)
# print("HG_sys^T == 0 ?", check_HG_zero(H, Gu))



# m, n = 7, 16
# # H = random_H_with_invertible_right(m, n, seed=42)
# # print("Initial random H:")
# # print(H)
# H = get_input_for_matrix()
# print(H)
# if H:
#     H = convert_str_input_into_matrix(H)
#     print("\nReducing to [A | I] using ROW OPS ONLY (logs on):")
#     H_std = to_standard_form_rows_only(H, verbose=True)

#     print("\nGenerator matrix G = [I | A^T]:")
#     G = generator_from_parity(H_std)
#     print(G)
# else:
#     print("Job done")
