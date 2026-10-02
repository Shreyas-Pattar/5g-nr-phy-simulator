"""
RRC pulse shaping + matched filtering, added on top of the
BPSK/QPSK AWGN simulation.

Chain:
bits -> map -> upsample -> RRC Tx filter -> AWGN
     -> RRC Rx matched filter -> downsample at correct timing -> demap -> bits
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

def q_function(x):
    return 0.5 * erfc(x / np.sqrt(2))

# ---------- mapper / demapper (same as step 1) ----------

def bpsk_mod(bits):
    return 1 - 2 * bits.astype(np.complex128)

def bpsk_demod(symbols):
    return (symbols.real < 0).astype(int)

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

# ---------- RRC filter ----------

def rrc_filter(beta, span, sps):
    """
    Root-raised-cosine filter impulse response.
    beta: roll-off factor (0 to 1)
    span: filter length in symbols (total length = span*sps + 1 samples)
    sps:  samples per symbol
    Returns h, normalized so that filtering (Tx conv Rx) has unit energy
    at the correct sampling instant.
    """
    N = span * sps
    t = (np.arange(-N/2, N/2 + 1)) / sps  # time in units of symbol periods
    h = np.zeros_like(t)

    for i, ti in enumerate(t):
        if ti == 0.0:
            h[i] = (1.0 - beta + 4 * beta / np.pi)
        elif beta != 0 and abs(abs(4 * beta * ti) - 1.0) < 1e-8:
            # special case avoids 0/0
            h[i] = (beta / np.sqrt(2)) * (
                ((1 + 2/np.pi) * np.sin(np.pi / (4 * beta))) +
                ((1 - 2/np.pi) * np.cos(np.pi / (4 * beta)))
            )
        else:
            num = np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            h[i] = num / den

    h = h / np.sqrt(np.sum(h ** 2))  # normalize filter energy to 1
    return h

def upsample(symbols, sps):
    up = np.zeros(len(symbols) * sps, dtype=np.complex128)
    up[::sps] = symbols
    return up

def awgn(signal, ebn0_db, bits_per_symbol, sps):
    """
    AWGN for the pulse-shaped signal.

    The RRC filter h is normalized to unit energy (sum(h^2)=1), which makes
    the Tx-filter -> matched-filter cascade have UNIT PEAK GAIN at the
    correct sampling instant (verified: an isolated unit symbol comes back
    out at amplitude 1.0). Because the filter doesn't rescale the signal
    peak, and its unit-energy normalization also means it doesn't rescale
    the noise variance at the sampling instant (Var[h * n] = sigma^2 *
    sum(h^2) = sigma^2), the per-sample noise variance needed is exactly
    the SAME N0/2 used in the plain (non-pulse-shaped) step-1 simulation.
    No extra sps factor belongs here -- that was the bug in the first
    version of this function.
    """
    ebn0_linear = 10 ** (ebn0_db / 10)
    eb = 1.0 / bits_per_symbol          # symbol energy Es = 1 (as in step 1)
    n0 = eb / ebn0_linear
    noise_std_per_dim = np.sqrt(n0 / 2)
    noise = noise_std_per_dim * (np.random.randn(*signal.shape)
                                  + 1j * np.random.randn(*signal.shape))
    return signal + noise

def simulate_ber_pulse_shaped(mod_fn, demod_fn, bits_per_symbol, ebn0_db,
                               sps=8, span=8, beta=0.35,
                               min_errors=100, max_bits=10_000_000, chunk=100_000):
    h = rrc_filter(beta, span, sps)
    filt_delay = span * sps // 2  # group delay of ONE filter, in samples

    total_bits = 0
    total_errors = 0
    while total_errors < min_errors and total_bits < max_bits:
        n_symbols = chunk // bits_per_symbol
        n_bits = n_symbols * bits_per_symbol
        bits = np.random.randint(0, 2, n_bits)
        symbols = mod_fn(bits)

        # Tx: upsample + pulse shape
        up = upsample(symbols, sps)
        tx = np.convolve(up, h)  # adds filt_delay samples at start (and end)

        # channel
        rx = awgn(tx, ebn0_db, bits_per_symbol, sps)

        # Rx: matched filter
        mf_out = np.convolve(rx, h)  # adds another filt_delay samples

        # total delay through BOTH filters = 2 * filt_delay
        # first valid symbol sample sits at index 2*filt_delay
        start = 2 * filt_delay
        sampled = mf_out[start: start + len(symbols) * sps: sps]

        rx_bits = demod_fn(sampled)
        errors = np.sum(bits != rx_bits)
        total_bits += n_bits
        total_errors += errors

    return total_errors / total_bits

def main():
    ebn0_range_db = np.arange(0, 11, 1)
    beta = 0.35
    sps = 8
    span = 8

    ber_bpsk = []
    ber_qpsk = []

    for ebn0_db in ebn0_range_db:
        b_bpsk = simulate_ber_pulse_shaped(bpsk_mod, bpsk_demod, 1, ebn0_db, sps, span, beta)
        b_qpsk = simulate_ber_pulse_shaped(qpsk_mod, qpsk_demod, 2, ebn0_db, sps, span, beta)
        ber_bpsk.append(b_bpsk)
        ber_qpsk.append(b_qpsk)
        print(f"Eb/N0 = {ebn0_db:2d} dB  BPSK BER = {b_bpsk:.3e}  QPSK BER = {b_qpsk:.3e}")

    ebn0_linear = 10 ** (ebn0_range_db / 10)
    ber_theory = q_function(np.sqrt(2 * ebn0_linear))

    plt.figure(figsize=(7, 5))
    plt.semilogy(ebn0_range_db, ber_bpsk, 'o-', label='BPSK (RRC pulse-shaped) simulated')
    plt.semilogy(ebn0_range_db, ber_qpsk, 's-', label='QPSK (RRC pulse-shaped) simulated')
    plt.semilogy(ebn0_range_db, ber_theory, 'k--', label='Theory Q(sqrt(2*Eb/N0))')
    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('Bit Error Rate')
    plt.title(f'BPSK/QPSK with RRC pulse shaping (beta={beta}) over AWGN')
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.savefig('ber_rrc_plot.png', dpi=150)
    plt.show()
    print("\nSaved plot to ber_rrc_plot.png")

    # ---- bonus sanity plot: eye diagram, proves ISI-free sampling ----
    bits = np.random.randint(0, 2, 2000)
    symbols = bpsk_mod(bits)
    h = rrc_filter(beta, span, sps)
    up = upsample(symbols, sps)
    tx = np.convolve(up, h)
    mf_out = np.convolve(tx, h)  # no noise, ideal channel, to show ISI-free eye
    filt_delay = span * sps // 2
    start = 2 * filt_delay

    plt.figure(figsize=(7, 5))
    trace_len = 2 * sps
    n_traces = 100
    for k in range(n_traces):
        seg_start = start - sps // 2 + k * sps
        seg = mf_out[seg_start: seg_start + trace_len]
        if len(seg) == trace_len:
            plt.plot(np.real(seg), color='steelblue', alpha=0.3)
    plt.title('Eye diagram after matched filter (no noise) — should be wide open')
    plt.xlabel('Sample index within symbol window')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.savefig('eye_diagram.png', dpi=150)
    plt.show()
    print("Saved eye diagram to eye_diagram.png")

if __name__ == "__main__":
    main()
