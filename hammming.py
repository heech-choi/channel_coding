import numpy as np
import matplotlib.pyplot as plt

# --- Repetition Code Functions (from previous code) ---
def repetition_encoder(data):
    return np.repeat(data, 3)

def bi_awgn_channel(encoded_data, snr_db, code_rate):
    signal = 1 - 2 * encoded_data
    snr_linear = 10**(snr_db / 10.0)
    noise_variance = 1 / (2 * code_rate * snr_linear)
    noise = np.sqrt(noise_variance) * np.random.randn(len(signal))
    return signal + noise

def repetition_decoder(received_signal):
    reshaped_signal = received_signal.reshape(-1, 3)
    summed_signal = np.sum(reshaped_signal, axis=1)
    decoded_data = (summed_signal <= 0).astype(int)
    return decoded_data

# --- Hamming (7,4) Code Functions (Newly Added) ---
def hamming_7_4_encoder(data):
    G = np.array([[1,0,0,0,1,1,0],[0,1,0,0,1,0,1],[0,0,1,0,0,1,1],[0,0,0,1,1,1,1]], dtype=int)
    if len(data) % 4 != 0:
        # 간단한 패딩 처리
        padding = 4 - (len(data) % 4)
        data = np.append(data, np.zeros(padding, dtype=int))
    data_blocks = data.reshape(-1, 4)
    encoded_blocks = np.dot(data_blocks, G) % 2
    return encoded_blocks.flatten()

def hamming_7_4_decoder(received_signal, original_len):
    H = np.array([[1,1,0,1,1,0,0],[1,0,1,1,0,1,0],[0,1,1,1,0,0,1]], dtype=int)
    hard_decisions = (received_signal <= 0).astype(int)
    received_blocks = hard_decisions.reshape(-1, 7)
    decoded_bits_list = []
    for block in received_blocks:
        syndrome = np.dot(block, H.T) % 2
        if np.all(syndrome == 0):
            decoded_bits_list.append(block[:4])
        else:
            error_pos = -1
            # syndrome과 H의 열을 비교하여 오류 위치 찾기 (효율적인 방식)
            for i in range(7):
                if np.array_equal(syndrome, H[:, i]):
                    error_pos = i
                    break
            corrected_block = np.copy(block)
            if error_pos != -1:
                corrected_block[error_pos] = 1 - corrected_block[error_pos]
            decoded_bits_list.append(corrected_block[:4])
    # 패딩을 고려하여 원본 길이만큼만 반환
    return np.array(decoded_bits_list).flatten()[:original_len]


def main():
    # --- Simulation Parameters ---
    # 정보 비트 길이는 4의 배수로 설정 (해밍 부호 때문)
    info_bits_len = 10000 * 4 
    snr_db_range = np.arange(0, 9, 1) # SNR range in dB
    
    ber_uncoded = []
    ber_repetition = []
    ber_hamming = []

    # --- Run Simulation for each SNR value ---
    for snr_db in snr_db_range:
        print(f"--- Simulating for SNR = {snr_db} dB ---")
        # Generate random source data
        source_bits = np.random.randint(0, 2, info_bits_len)
        
        # --- 1. Uncoded BPSK Simulation ---
        signal_uncoded = 1 - 2 * source_bits
        received_uncoded = bi_awgn_channel(source_bits, snr_db, code_rate=1.0)
        decoded_uncoded = (received_uncoded <= 0).astype(int)
        ber_uncoded.append(np.sum(source_bits != decoded_uncoded) / info_bits_len)

        # --- 2. Repetition Code (3,1) Simulation ---
        encoded_rep = repetition_encoder(source_bits)
        received_rep = bi_awgn_channel(encoded_rep, snr_db, code_rate=1/3)
        decoded_rep = repetition_decoder(received_rep)
        ber_repetition.append(np.sum(source_bits != decoded_rep) / info_bits_len)
        
        # --- 3. Hamming Code (7,4) Simulation ---
        encoded_ham = hamming_7_4_encoder(source_bits)
        received_ham = bi_awgn_channel(encoded_ham, snr_db, code_rate=4/7)
        # 디코더에 원본 데이터 길이를 전달하여 패딩 처리
        decoded_ham = hamming_7_4_decoder(received_ham, info_bits_len)
        ber_hamming.append(np.sum(source_bits != decoded_ham) / info_bits_len)

    # --- Plotting the Results ---
    plt.figure(figsize=(12, 8))
    plt.semilogy(snr_db_range, ber_uncoded, marker='s', linestyle=':', label='Uncoded BPSK (Simulated)')
    plt.semilogy(snr_db_range, ber_repetition, marker='o', linestyle='--', label='Repetition Code (3,1) (Simulated)')
    plt.semilogy(snr_db_range, ber_hamming, marker='^', linestyle='-', label='Hamming Code (7,4) (Simulated)')
    
    plt.title('BER Performance Comparison of Channel Codes')
    plt.xlabel('$E_b/N_0$ (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.legend()
    plt.grid(True, which="both")
    plt.ylim(1e-5, 1.0)
    plt.xlim(0, 8)
    plt.show()

main()