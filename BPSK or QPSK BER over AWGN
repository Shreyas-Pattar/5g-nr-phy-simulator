"""
BPSK / QPSK BER over AWGN
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

def q_function(x):
    """Gaussian tail probability Q(x) = 0.5*erfc(x/sqrt(2))."""
    return 0.5 * erfc(x / np.sqrt(2))

def bpsk_mod(bits):
    # 0 -> +1, 1 -> -1
    return 1 - 2 * bits.astype(np.complex128)

def bpsk_demod(rx_symbols):
    # decide based on sign of real part
    return (rx_symbols.real < 0).astype(int)

def qpsk_mod(bits):
    """
    bits: 1D array, length must be even.
    Two bits -> one symbol, Gray-coded, unit energy.
    (b0,b1) = (0,0) -> (+1+1j)/sqrt2
              (0,1) -> (+1-1j)/sqrt2
              (1,0) -> (-1+1j)/sqrt2
              (1,1) -> (-1-1j)/sqrt2
    """
    b0 = bits[0::2]
    b1 = bits[1::2]
    i = 1 - 2 * b0   # +1 if b0=0, -1 if b0=1
    q = 1 - 2 * b1
    symbols = (i + 1j * q) / np.sqrt(2)
    return symbols

def qpsk_demod(rx_symbols):
    b0 = (rx_symbols.real < 0).astype(int)
    b1 = (rx_symbols.imag < 0).astype(int)
    bits = np.empty(2 * len(rx_symbols), dtype=int)
    bits[0::2] = b0
    bits[1::2] = b1
    return bits

def awgn(symbols, ebn0_db, bits_per_symbol):
    """
    Add complex AWGN scaled from Eb/N0 (dB).
    Symbol energy Es = bits_per_symbol * Eb  (we normalize Es=1 in the mapper,
    so Eb = 1/bits_per_symbol).
    N0 = Eb / (10^(Eb/N0_dB / 10))
    Noise variance per complex dimension = N0/2 (real and imag each get N0/2,
    so total noise power = N0).
    """
    ebn0_linear = 10 ** (ebn0_db / 10)
    eb = 1.0 / bits_per_symbol         # since Es (symbol energy) = 1
    n0 = eb / ebn0_linear
    noise_std_per_dim = np.sqrt(n0 / 2)
    noise = noise_std_per_dim * (np.random.randn(*symbols.shape)
                                  + 1j * np.random.randn(*symbols.shape))
    return symbols + noise

def simulate_ber(mod_fn, demod_fn, bits_per_symbol, ebn0_db, min_errors=100, max_bits=10_000_000, chunk=200_000):
    """Run until we collect >= min_errors bit errors, or hit max_bits."""
    total_bits = 0
    total_errors = 0
    while total_errors < min_errors and total_bits < max_bits:
        n_symbols = chunk // bits_per_symbol
        n_bits = n_symbols * bits_per_symbol
        bits = np.random.randint(0, 2, n_bits)
        symbols = mod_fn(bits)
        rx = awgn(symbols, ebn0_db, bits_per_symbol)
        rx_bits = demod_fn(rx)
        errors = np.sum(bits != rx_bits)
        total_bits += n_bits
        total_errors += errors
    return total_errors / total_bits

def main():
    ebn0_range_db = np.arange(0, 11, 1)

    ber_bpsk_sim = []
    ber_qpsk_sim = []

    for ebn0_db in ebn0_range_db:
        ber_bpsk_sim.append(simulate_ber(bpsk_mod, bpsk_demod, 1, ebn0_db))
        ber_qpsk_sim.append(simulate_ber(qpsk_mod, qpsk_demod, 2, ebn0_db))
        print(f"Eb/N0 = {ebn0_db:2d} dB  BPSK BER = {ber_bpsk_sim[-1]:.3e}  "
              f"QPSK BER = {ber_qpsk_sim[-1]:.3e}")

    ebn0_linear = 10 ** (ebn0_range_db / 10)
    ber_theory = q_function(np.sqrt(2 * ebn0_linear))  # same formula, per-bit, for BPSK and QPSK

    plt.figure(figsize=(7, 5))
    plt.semilogy(ebn0_range_db, ber_bpsk_sim, 'o-', label='BPSK simulated')
    plt.semilogy(ebn0_range_db, ber_qpsk_sim, 's-', label='QPSK simulated')
    plt.semilogy(ebn0_range_db, ber_theory, 'k--', label='Theory Q(sqrt(2*Eb/N0))')
    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('Bit Error Rate')
    plt.title('BPSK / QPSK over AWGN: Simulated vs Theoretical BER')
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.savefig('ber_awgn_plot.png', dpi=150)
    plt.show()
    print("\nSaved plot to ber_awgn_plot.png")

if __name__ == "__main__":
    main()
