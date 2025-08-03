import numpy as np
import matplotlib.pyplot as plt

def repetition_encoder(data):
    """
    Encodes data by repeating each bit 3 times.
    Example: [1, 0] -> [1, 1, 1, 0, 0, 0]
    """
    return np.repeat(data, 3)

def bi_awgn_channel(encoded_data, snr_db, code_rate):
    """
    Simulates a BI-AWGN channel.
    Converts bits to BPSK symbols (0 -> 1, 1 -> -1), adds Gaussian noise,
    and returns the noisy signal.
    """
    # Convert bits to BPSK symbols
    signal = 1 - 2 * encoded_data
    # Calculate linear SNR from dB
    snr_linear = 10**(snr_db / 10.0)
    # Calculate noise variance based on code rate and SNR
    noise_variance = 1 / (2 * code_rate * snr_linear)
    # Generate Gaussian noise
    noise = np.sqrt(noise_variance) * np.random.randn(len(signal))
    # Return the noisy signal
    return signal + noise

def repetition_decoder(received_signal):
    """
    Decodes the received signal using majority voting.
    It reshapes the signal into blocks of 3 and decides the bit based on the sum.
    Sum > 0 -> 0, Sum <= 0 -> 1 (since 0 is mapped to +1 and 1 to -1)
    """
    # Reshape the signal into chunks of 3
    reshaped_signal = received_signal.reshape(-1, 3)
    # Sum each chunk
    summed_signal = np.sum(reshaped_signal, axis=1)
    # Majority vote: if sum > 0, it's likely a 0 (mapped to +1). Otherwise, it's a 1.
    decoded_data = (summed_signal <= 0).astype(int)
    return decoded_data

def main():
    """
    Main function to run the simulation and plot the results.
    """
    # --- Simulation Parameters ---
    info_bits_len = 10000  # Number of information bits to transmit
    snr_db_range = np.arange(0, 10, 1) # SNR range in dB
    
    # The rate of our code is 1/3 since we send 3 bits for every 1 info bit
    code_rate = 1/3
    
    # Array to store Bit Error Rate results
    ber_results = []

    print("Running simulation...")
    # --- Run Simulation for each SNR value ---
    for snr_db in snr_db_range:
        # Generate random source data
        source_bits = np.random.randint(0, 2, info_bits_len)
        
        # 1. Encode the data
        encoded_bits = repetition_encoder(source_bits)
        
        # 2. Pass through the AWGN channel
        received_signal = bi_awgn_channel(encoded_bits, snr_db, code_rate)
        
        # 3. Decode the received signal
        decoded_bits = repetition_decoder(received_signal)
        
        # 4. Calculate the number of bit errors
        num_errors = np.sum(source_bits != decoded_bits)
        
        # 5. Calculate the Bit Error Rate (BER)
        ber = num_errors / info_bits_len
        ber_results.append(ber)
        print(f"SNR: {snr_db} dB, BER: {ber:.6f}")

    # --- Plotting the Results ---
    plt.figure(figsize=(10, 7))
    plt.semilogy(snr_db_range, ber_results, marker='o', linestyle='-', label='3-Repetition Code')
    
    # Optional: Plot theoretical uncoded BPSK for comparison
    from scipy.special import erfc
    snr_linear_range = 10**(snr_db_range / 10.0)
    ber_uncoded_theoretical = 0.5 * erfc(np.sqrt(snr_linear_range))
    plt.semilogy(snr_db_range, ber_uncoded_theoretical, marker='x', linestyle='--', label='Uncoded BPSK (Theoretical)')

    plt.title('BER Performance of 3-Repetition Code in BI-AWGN Channel')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.legend()
    plt.grid(True, which="both")
    plt.ylim(1e-5, 1.0)
    plt.show()

if __name__ == '__main__':
    # You might need to install scipy for the theoretical curve: pip install scipy
    # If you don't have it, you can comment out the theoretical plot lines.
    main()