"""
OFDM channel ESTIMATION.

Same OFDM/CP/multipath setup, but now the receiver does NOT
know H. It estimates H from pilot subcarriers using:
  - LS (least squares) + linear interpolation  -- the naive baseline
  - MMSE (Wiener filter) using known channel frequency correlation
      (derived from the L-tap uniform power-delay-profile assumption)

All three (perfect CSI, LS, MMSE) are run side by side so the gap between
them is the actual result to report.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

def q_function(x):
    return 0.5 * erfc(x / np.sqrt(2))

def ber_awgn_theory(ebn0_linear):
    return q_function(np.sqrt(2 * ebn0_linear))

def ber_rayleigh_theory(ebn0_linear):
    g = ebn0_linear
    return 0.5 * (1 - np.sqrt(g / (1 + g)))

# ---------- QPSK ----------

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
    """iid CN(0,1/L) taps -- fixed EXPECTED energy 1, realized energy
    fluctuates naturally (see step-3a note on why forcing unit realized
    energy is wrong)."""
    return (np.random.randn(L) + 1j * np.random.randn(L)) / np.sqrt(2 * L)

# ---------- OFDM ----------

def ofdm_mod(X, N, cp_len):
    x = np.fft.ifft(X, n=N)
    cp = x[-cp_len:]
    return np.concatenate([cp, x])

def ofdm_strip_and_fft(y_full, N, cp_len):
    y = y_full[cp_len: cp_len + N]
    return np.fft.fft(y, n=N)

# ---------- channel estimation ----------

def channel_freq_corr(k1, k2, N, L):
    """R[i,j] = E[H(k1_i) H*(k2_j)] for an L-tap channel with iid taps of
    variance 1/L each (uniform power-delay profile). Closed form:
    R(k1,k2) = (1/L) * sum_{l=0}^{L-1} exp(-j*2*pi*(k1-k2)*l/N)."""
    diff = k1[:, None] - k2[None, :]
    R = np.zeros(diff.shape, dtype=complex)
    for l in range(L):
        R += np.exp(-1j * 2 * np.pi * diff * l / N)
    return R / L

def ls_estimate_with_interp(Y, pilot_idx, pilot_val, N):
    H_ls_pilot = Y[pilot_idx] / pilot_val
    all_idx = np.arange(N)
    H_real = np.interp(all_idx, pilot_idx, H_ls_pilot.real)
    H_imag = np.interp(all_idx, pilot_idx, H_ls_pilot.imag)
    return H_real + 1j * H_imag, H_ls_pilot

def mmse_estimate(H_ls_pilot, pilot_idx, N, L, n0):
    R_pp = channel_freq_corr(pilot_idx, pilot_idx, N, L)
    R_fp = channel_freq_corr(np.arange(N), pilot_idx, N, L)
    W = R_fp @ np.linalg.inv(R_pp + n0 * np.eye(len(pilot_idx)))
    return W @ H_ls_pilot

# ---------- simulation ----------

def simulate(ebn0_db, N=64, cp_len=16, L=8, pilot_spacing=4,
             min_errors=200, max_blocks=200_000):
    ebn0_linear = 10 ** (ebn0_db / 10)
    bits_per_symbol = 2
    eb = 1.0 / bits_per_symbol
    n0 = eb / ebn0_linear
    noise_std_per_dim = np.sqrt((n0 / N) / 2)

    pilot_idx = np.arange(0, N, pilot_spacing)
    data_idx = np.setdiff1d(np.arange(N), pilot_idx)
    pilot_val = (1 + 1j) / np.sqrt(2)   # known unit-energy pilot symbol

    counts = {'perfect': [0, 0], 'ls': [0, 0], 'mmse': [0, 0]}  # [errors, bits]

    blocks = 0
    while min(counts[k][0] for k in counts) < min_errors and blocks < max_blocks:
        blocks += 1

        n_data_bits = len(data_idx) * bits_per_symbol
        bits = np.random.randint(0, 2, n_data_bits)
        data_symbols = qpsk_mod(bits)

        X = np.zeros(N, dtype=complex)
        X[pilot_idx] = pilot_val
        X[data_idx] = data_symbols

        tx = ofdm_mod(X, N, cp_len)
        h = random_multipath_channel(L)
        y_full = np.convolve(tx, h)
        noise = noise_std_per_dim * (np.random.randn(*y_full.shape)
                                      + 1j * np.random.randn(*y_full.shape))
        y_full = y_full + noise

        Y = ofdm_strip_and_fft(y_full, N, cp_len)
        H_true = np.fft.fft(h, n=N)

        # --- perfect CSI ---
        X_hat = Y / H_true
        rx_bits = qpsk_demod(X_hat[data_idx])
        counts['perfect'][0] += np.sum(bits != rx_bits)
        counts['perfect'][1] += n_data_bits

        # --- LS + interpolation ---
        H_ls_full, H_ls_pilot = ls_estimate_with_interp(Y, pilot_idx, pilot_val, N)
        X_hat_ls = Y / H_ls_full
        rx_bits_ls = qpsk_demod(X_hat_ls[data_idx])
        counts['ls'][0] += np.sum(bits != rx_bits_ls)
        counts['ls'][1] += n_data_bits

        # --- MMSE ---
        H_mmse_full = mmse_estimate(H_ls_pilot, pilot_idx, N, L, n0)
        X_hat_mmse = Y / H_mmse_full
        rx_bits_mmse = qpsk_demod(X_hat_mmse[data_idx])
        counts['mmse'][0] += np.sum(bits != rx_bits_mmse)
        counts['mmse'][1] += n_data_bits

    return {k: counts[k][0] / counts[k][1] for k in counts}

def main():
    ebn0_range_db = np.arange(0, 26, 2)
    N, cp_len, L = 64, 16, 8

    results = {'perfect': [], 'ls': [], 'mmse': []}
    for ebn0_db in ebn0_range_db:
        r = simulate(ebn0_db, N=N, cp_len=cp_len, L=L)
        for k in results:
            results[k].append(r[k])
        print(f"Eb/N0 = {ebn0_db:2d} dB  perfect={r['perfect']:.3e}  "
              f"LS={r['ls']:.3e}  MMSE={r['mmse']:.3e}")

    ebn0_linear = 10 ** (ebn0_range_db / 10)
    ber_rayleigh = ber_rayleigh_theory(ebn0_linear)

    plt.figure(figsize=(7.5, 5.5))
    plt.semilogy(ebn0_range_db, results['perfect'], 'o-', label='Perfect CSI (genie)')
    plt.semilogy(ebn0_range_db, results['mmse'], '^-', label='MMSE estimation')
    plt.semilogy(ebn0_range_db, results['ls'], 's-', label='LS + linear interpolation')
    plt.semilogy(ebn0_range_db, ber_rayleigh, 'k--', label='Rayleigh theory (perfect CSI ref)')
    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('Bit Error Rate')
    plt.title(f'OFDM channel estimation: perfect CSI vs LS vs MMSE (N={N}, L={L})')
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.savefig('ofdm_channel_estimation_ber.png', dpi=150)
    plt.show()
    print("\nSaved plot to ofdm_channel_estimation_ber.png")

if __name__ == "__main__":
    main()
