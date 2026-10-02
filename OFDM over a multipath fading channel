"""
OFDM over a multipath fading channel, with PERFECT channel
knowledge (genie CSI) and one-tap zero-forcing equalization per subcarrier.

This isolates the OFDM mechanics (IFFT/CP/FFT/per-subcarrier equalization)
from channel ESTIMATION, which comes in step 3b (LS/MMSE pilots).

Validation target: BER should match the closed-form Rayleigh flat-fading
BER curve, NOT the AWGN curve -- fading is significantly worse than AWGN
at the same average Eb/N0, and that gap is the whole point of this step.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

def q_function(x):
    return 0.5 * erfc(x / np.sqrt(2))

def ber_awgn_theory(ebn0_linear):
    return q_function(np.sqrt(2 * ebn0_linear))

def ber_rayleigh_theory(ebn0_linear):
    """Closed-form average BER for BPSK/QPSK over Rayleigh flat fading,
    coherent detection, perfect CSI."""
    g = ebn0_linear
    return 0.5 * (1 - np.sqrt(g / (1 + g)))

# ---------- QPSK mapper / demapper (unit energy per symbol) ----------

def qpsk_mod(bits):
    b0 = bits[0::2]
    b1 = bits[1::2]
    i = 1 - 2 * b0
    q = 1 - 2 * b1
    return (i + 1j * q) / np.sqrt(2)

def qpsk_demod(symbols):
    b0 = (symbols.real < 0).astype(int)
    b1 = (symbols.imag < 0).astype(int)
    bits = np.empty(2 * len(symbols), dtype=int)
    bits[0::2] = b0
    bits[1::2] = b1
    return bits

# ---------- channel ----------

def random_multipath_channel(L):
    """L-tap Rayleigh multipath channel. Each tap is iid CN(0, 1/L), so the
    EXPECTED total energy is 1 (sum of L taps each with variance 1/L).

    Important: we do NOT force-normalize each individual realization to
    have exactly unit energy (i.e. do NOT divide by its own realized norm).
    Doing that was a real bug caught by comparing against theory -- forcing
    a fixed total energy per block removes the natural realization-to-
    realization energy fluctuation that a true Rayleigh channel has, which
    suppresses deep fades and makes the simulated BER come out too good
    (systematically ~10-15% below the closed-form curve, confirmed with
    tight statistics). Leaving each tap's variance fixed at 1/L (letting
    the total energy fluctuate naturally around 1) is what makes each
    subcarrier's H[k] = FFT(h) exactly complex Gaussian, i.e. genuinely
    Rayleigh-distributed in magnitude, matching the theory formula's
    assumption."""
    h = (np.random.randn(L) + 1j * np.random.randn(L)) / np.sqrt(2 * L)
    return h

# ---------- OFDM ----------

def ofdm_mod(X, N, cp_len):
    """X: N frequency-domain symbols -> time domain with cyclic prefix."""
    x = np.fft.ifft(X, n=N)          # numpy ifft includes the 1/N factor
    cp = x[-cp_len:]
    return np.concatenate([cp, x])

def ofdm_strip_and_fft(y_full, N, cp_len):
    """Take the N-sample window starting right after the CP. As long as
    cp_len >= L-1 (channel length - 1), this window equals the CIRCULAR
    convolution of the transmitted block and the channel, which is exactly
    what makes per-subcarrier division work after the FFT."""
    y = y_full[cp_len: cp_len + N]
    return np.fft.fft(y, n=N)

def simulate_ofdm_ber(ebn0_db, N=64, cp_len=16, L=8, bits_per_symbol=2,
                       min_errors=200, max_symbols_blocks=200_000):
    ebn0_linear = 10 ** (ebn0_db / 10)
    eb = 1.0 / bits_per_symbol           # Es = 1 per subcarrier symbol
    n0 = eb / ebn0_linear
    # derived in the explanation: time-domain noise variance per complex
    # sample must be N0/N so that after the (unnormalized) receive FFT,
    # each subcarrier's noise variance is back to N0 -- matching the same
    # N0 convention used in steps 1 and 2.
    noise_var_per_sample = n0 / N
    noise_std_per_dim = np.sqrt(noise_var_per_sample / 2)

    total_bits = 0
    total_errors = 0
    blocks = 0

    while total_errors < min_errors and blocks < max_symbols_blocks:
        blocks += 1
        bits = np.random.randint(0, 2, N * bits_per_symbol)
        X = qpsk_mod(bits)                       # N unit-energy QPSK symbols

        tx = ofdm_mod(X, N, cp_len)               # time domain + CP

        h = random_multipath_channel(L)           # new channel per OFDM symbol
        y_full = np.convolve(tx, h)                # linear conv (physical channel)

        noise = noise_std_per_dim * (np.random.randn(*y_full.shape)
                                      + 1j * np.random.randn(*y_full.shape))
        y_full = y_full + noise

        Y = ofdm_strip_and_fft(y_full, N, cp_len)

        H = np.fft.fft(h, n=N)                     # GENIE: exact channel, known to Rx

        X_hat = Y / H                               # one-tap zero-forcing equalizer

        rx_bits = qpsk_demod(X_hat)
        errors = np.sum(bits != rx_bits)
        total_bits += len(bits)
        total_errors += errors

    return total_errors / total_bits

def main():
    ebn0_range_db = np.arange(0, 26, 2)
    N, cp_len, L = 64, 16, 8

    ber_sim = []
    for ebn0_db in ebn0_range_db:
        b = simulate_ofdm_ber(ebn0_db, N=N, cp_len=cp_len, L=L)
        ber_sim.append(b)
        print(f"Eb/N0 = {ebn0_db:2d} dB  OFDM(known-H,ZF) BER = {b:.3e}")

    ebn0_linear = 10 ** (ebn0_range_db / 10)
    ber_rayleigh = ber_rayleigh_theory(ebn0_linear)
    ber_awgn = ber_awgn_theory(ebn0_linear)

    plt.figure(figsize=(7, 5))
    plt.semilogy(ebn0_range_db, ber_sim, 'o-', label='OFDM simulated (known H, ZF)')
    plt.semilogy(ebn0_range_db, ber_rayleigh, 'k--', label='Rayleigh flat-fading theory')
    plt.semilogy(ebn0_range_db, ber_awgn, 'g:', label='AWGN theory (reference, no fading)')
    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('Bit Error Rate')
    plt.title(f'OFDM (N={N}, CP={cp_len}, L={L}-tap channel), perfect CSI, ZF equalizer')
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.savefig('ofdm_known_channel_ber.png', dpi=150)
    plt.show()
    print("\nSaved plot to ofdm_known_channel_ber.png")

if __name__ == "__main__":
    main()
