import numpy as np
import matplotlib.pyplot as plt
from pyldpc import make_ldpc, encode, decode, get_message
from hermes_py.core import Turbo

def bi_awgn_channel(signal, snr_db, code_rate):
    """
    Simulates a Binary Input Additive White Gaussian Noise (BI-AWGN) channel.

    Args:
        signal (numpy.ndarray): The input signal (0s and 1s).
        snr_db (float): The signal-to-noise ratio in decibels.
        code_rate (float): The rate of the channel code.

    Returns:
        numpy.ndarray: The signal with added noise.
    """
    # Convert signal to BPSK: 0 -> 1, 1 -> -1
    bpsk_signal = 1 - 2 * signal
    # Calculate linear SNR
    snr_linear = 10**(snr_db / 10.0)
    # Calculate noise variance
    noise_variance = 1 / (2 * code_rate * snr_linear)
    # Generate Gaussian noise
    noise = np.sqrt(noise_variance) * np.random.randn(*bpsk_signal.shape)
    # Add noise to the signal
    received_signal = bpsk_signal + noise
    return received_signal

def main():
    """
    Main function to test and compare channel codes.
    """
    # --- Simulation Parameters ---
    info_bits_len = 100  # Length of the information bits
    snr_range_db = np.arange(0, 6, 0.5)  # SNR range in dB

    # --- Initialize arrays to store Bit Error Rates ---
    ber_ldpc = []
    ber_turbo = []

    # --- LDPC Code Parameters ---
    n_ldpc = 200  # Codeword length
    d_v = 2       # Number of 1s per column in H
    d_c = 4       # Number of 1s per row in H
    
    # Generate LDPC code matrices
    H, G = make_ldpc(n_ldpc, d_v, d_c, systematic=True, sparse=True)
    ldpc_code_rate = G.shape[1] / n_ldpc

    # --- Turbo Code Parameters ---
    turbo_code = Turbo(bit_block_size=info_bits_len, poly_a=0o13, poly_b=0o15, num_iterations=10)
    turbo_code_rate = turbo_code.rate

    # --- Run Simulation for each SNR value ---
    for snr_db in snr_range_db:
        # Generate random information bits
        info_bits = np.random.randint(0, 2, info_bits_len)

        # --- LDPC Simulation ---
        encoded_ldpc = encode(G, info_bits)
        received_ldpc = bi_awgn_channel(encoded_ldpc, snr_db, ldpc_code_rate)
        decoded_ldpc = decode(H, received_ldpc, snr_db)
        decoded_message_ldpc = get_message(G, decoded_ldpc)
        ldpc_errors = np.sum(info_bits != decoded_message_ldpc)
        ber_ldpc.append(ldpc_errors / info_bits_len)

        # --- Turbo Code Simulation ---
        encoded_turbo = turbo_code.encode(info_bits)
        received_turbo = bi_awgn_channel(encoded_turbo, snr_db, turbo_code_rate)
        # Convert received signal to log-likelihood ratios (LLRs) for Turbo decoder
        # A simple approximation for BPSK is LLR = 2*received_signal/noise_variance
        snr_linear = 10**(snr_db / 10.0)
        noise_variance = 1 / (2 * turbo_code_rate * snr_linear)
        llr_turbo = 2 * received_turbo / noise_variance
        decoded_turbo = turbo_code.decode(llr_turbo)
        turbo_errors = np.sum(info_bits != decoded_turbo)
        ber_turbo.append(turbo_errors / info_bits_len)

    # --- Plotting the Results ---
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_range_db, ber_ldpc, 'o-', label='LDPC')
    plt.semilogy(snr_range_db, ber_turbo, 's-', label='Turbo')
    plt.grid(True, which='both')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title('Performance of Channel Codes in BI-AWGN Channel')
    plt.legend()
    plt.ylim(1e-5, 1)
    plt.show()

if __name__ == '__main__':
    main()