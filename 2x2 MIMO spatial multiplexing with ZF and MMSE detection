"""
2x2 MIMO spatial multiplexing with ZF and MMSE detection.

Two independent QPSK streams transmitted simultaneously from 2 Tx
antennas, received on 2 Rx antennas:
    y = H x + n
H is 2x2, flat Rayleigh fading (i.i.d. CN(0,1) entries), new realization
per channel use (no multipath/OFDM here -- MIMO kept isolated for a clean
first validation; combine with OFDM later if you want MIMO-OFDM).

Parameterization note (read this before comparing to SISO): Eb/N0 here
is defined PER STREAM, exactly like every earlier SISO plot in this
project (Es=1 per stream symbol, Eb=Es/bits_per_symbol). This is the
standard way to show the "MIMO ZF/MMSE noise enhancement" story: at the
SAME per-stream SNR a SISO link would see, 2x2 MIMO detection does worse,
because separating two simultaneous streams costs you noise amplification
(ZF) or at least SNR-reducing regularization (MMSE). It is NOT a per-
total-system-energy comparison -- that would be a different (fairer, but
different) accounting of MIMO's doubled throughput, left as a documented
simplification.

Detectors:
  ZF:   x_hat = H^-1 y                           (perfectly cancels
        inter-stream interference, but amplifies noise when H is
        near-singular -- the two Tx paths look too similar to the Rx
        antennas to tell apart)
  MMSE: x_hat = (H^H H + (N0/Es) I)^-1 H^H y      (balances interference
        cancellation against noise amplification)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

def q_function(x):
    return 0.5 * erfc(x / np.sqrt(2))

def ber_rayleigh_theory_siso(ebn0_linear):
    """Closed-form BER, single-antenna Rayleigh flat fading, coherent
    detection, perfect CSI -- the SISO reference curve from step 3a."""
    g = ebn0_linear
    return 0.5 * (1 - np.sqrt(g / (1 + g)))

# ---------- QPSK (same convention used throughout this project) ----------

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

# ---------- MIMO channel ----------

def random_mimo_channel():
    """2x2, i.i.d. CN(0,1) entries -- standard Rayleigh MIMO model."""
    return (np.random.randn(2, 2) + 1j * np.random.randn(2, 2)) / np.sqrt(2)

# ---------- simulation ----------

def simulate_mimo_ber(ebn0_db, detector='zf', min_errors=300, max_blocks=500_000):
    bits_per_symbol = 2  # QPSK
    eb = 1.0 / bits_per_symbol       # Es = 1 per stream, same convention as all earlier steps
    ebn0_linear = 10 ** (ebn0_db / 10)
    n0 = eb / ebn0_linear
    noise_std_per_dim = np.sqrt(n0 / 2)

    total_bits = 0
    total_errors = 0
    blocks = 0

    while total_errors < min_errors and blocks < max_blocks:
        blocks += 1
        bits1 = np.random.randint(0, 2, 2)   # 2 bits -> 1 QPSK symbol, stream 1
        bits2 = np.random.randint(0, 2, 2)   # stream 2
        x1 = qpsk_mod(bits1)[0]
        x2 = qpsk_mod(bits2)[0]
        x = np.array([x1, x2])

        H = random_mimo_channel()
        noise = noise_std_per_dim * (np.random.randn(2) + 1j * np.random.randn(2))
        y = H @ x + noise

        if detector == 'zf':
            x_hat = np.linalg.inv(H) @ y
        elif detector == 'mmse':
            Hh = H.conj().T
            x_hat = np.linalg.inv(Hh @ H + n0 * np.eye(2)) @ Hh @ y
        else:
            raise ValueError(detector)

        rx_bits1 = qpsk_demod(np.array([x_hat[0]]))
        rx_bits2 = qpsk_demod(np.array([x_hat[1]]))

        errors = np.sum(bits1 != rx_bits1) + np.sum(bits2 != rx_bits2)
        total_bits += 4   # 2 bits/stream * 2 streams
        total_errors += errors

    return total_errors / total_bits

def main():
    ebn0_range_db = np.arange(0, 26, 2)

    ber_zf, ber_mmse = [], []
    for ebn0_db in ebn0_range_db:
        b_zf = simulate_mimo_ber(ebn0_db, detector='zf')
        b_mmse = simulate_mimo_ber(ebn0_db, detector='mmse')
        ber_zf.append(b_zf)
        ber_mmse.append(b_mmse)
        print(f"Eb/N0 = {ebn0_db:2d} dB  ZF BER = {b_zf:.3e}  MMSE BER = {b_mmse:.3e}")

    ebn0_linear = 10 ** (ebn0_range_db / 10)
    ber_siso_rayleigh = ber_rayleigh_theory_siso(ebn0_linear)

    plt.figure(figsize=(7.5, 5.5))
    plt.semilogy(ebn0_range_db, ber_zf, 'o-', label='2x2 MIMO ZF detection')
    plt.semilogy(ebn0_range_db, ber_mmse, '^-', label='2x2 MIMO MMSE detection')
    plt.semilogy(ebn0_range_db, ber_siso_rayleigh, 'k--',
                 label='SISO Rayleigh theory (reference, no spatial mux)')
    plt.xlabel('Eb/N0 per stream (dB)')
    plt.ylabel('Bit Error Rate')
    plt.title('2x2 MIMO spatial multiplexing: ZF vs MMSE detection')
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.savefig('mimo_2x2_ber.png', dpi=150)
    plt.show()
    print("\nSaved plot to mimo_2x2_ber.png")

if __name__ == "__main__":
    main()
