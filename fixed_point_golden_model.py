%%writefile fixed_point_golden_model.py
"""
Fixed-point (Q1.15) golden model for QPSK + RRC pulse shaping + matched
filter, NO noise (deterministic chain only -- see module docstring in the
companion C++ file for why noise is excluded from the bit-exact check).

This script:
  1. Builds the same RRC filter used in step 2, quantizes it to Q1.15.
  2. Builds a deterministic (not random) QPSK test sequence, quantizes it.
  3. Runs the ENTIRE chain in pure integer (Q1.15) arithmetic, exactly as
     the C++ port must, including the exact rounding/shift convention.
  4. Writes coefficients, input symbols, and the final output to text
     files the C++ program reads/writes, so the two can be diffed for
     an exact bit-for-bit match.
"""

import numpy as np

Q15_SCALE = 32768  # 2^15
Q15_MAX = 32767
Q15_MIN = -32768

def to_q15(x):
    """Quantize a float array to Q1.15 (round to nearest, saturate)."""
    scaled = np.round(x * Q15_SCALE).astype(np.int64)
    return np.clip(scaled, Q15_MIN, Q15_MAX).astype(np.int32)

def q15_mul_round_shift(a, b):
    """
    Q1.15 x Q1.15 -> Q1.15 multiply.
    a, b: int32 arrays/scalars holding Q1.15 values.
    Product is Q2.30 (32x32->64-bit intermediate); round and shift back
    down to Q1.15.

    Rounding convention (MUST match the C++ side exactly):
        shifted = (product + (1 << 14)) >> 15
    then saturate to int16 range. The >> here is an ARITHMETIC right
    shift (floor-based) on a two's-complement value -- Python's integer
    >> already does this (floor division by a power of 2), and so does
    standard C++ on a signed 64-bit int on every mainstream compiler.
    If this convention differs between the two languages, outputs will
    NOT match bit-exact -- that mismatch is exactly the kind of bug this
    whole exercise is designed to catch.
    """
    product = a.astype(np.int64) * b.astype(np.int64)
    rounded = (product + (1 << 14)) >> 15
    return np.clip(rounded, Q15_MIN, Q15_MAX).astype(np.int32)

def rrc_filter_float(beta, span, sps):
    N = span * sps
    t = (np.arange(-N / 2, N / 2 + 1)) / sps
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti == 0.0:
            h[i] = (1.0 - beta + 4 * beta / np.pi)
        elif beta != 0 and abs(abs(4 * beta * ti) - 1.0) < 1e-8:
            h[i] = (beta / np.sqrt(2)) * (
                ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))) +
                ((1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
            )
        else:
            num = np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            h[i] = num / den
    h = h / np.sqrt(np.sum(h ** 2))  # same normalization as step 2
    return h

def qpsk_mod_deterministic(bits):
    b0 = bits[0::2]
    b1 = bits[1::2]
    i = 1 - 2 * b0.astype(float)
    q = 1 - 2 * b1.astype(float)
    return (i + 1j * q) / np.sqrt(2)

def fixed_point_fir(x_q15, h_q15):
    """Full convolution, Q1.15 in, Q1.15 out, integer arithmetic only."""
    N, M = len(x_q15), len(h_q15)
    y = np.zeros(N + M - 1, dtype=np.int32)
    # accumulate in int64 across ALL taps for one output sample, THEN
    # round/shift/saturate ONCE -- matches standard fixed-point FIR
    # practice (accumulator stays wide, quantization happens at output)
    for n in range(N + M - 1):
        acc = np.int64(0)
        for k in range(M):
            xi = n - k
            if 0 <= xi < N:
                acc += np.int64(x_q15[xi]) * np.int64(h_q15[k])
        rounded = (acc + (1 << 14)) >> 15
        y[n] = np.clip(rounded, Q15_MIN, Q15_MAX)
    return y

def main():
    beta, span, sps = 0.35, 8, 8
    h_float = rrc_filter_float(beta, span, sps)
    h_q15 = to_q15(h_float)

    # deterministic test pattern -- NOT random, so the C++ side can use
    # the exact same fixed input without needing a shared RNG
    test_bits = np.array([0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0], dtype=int)
    symbols = qpsk_mod_deterministic(test_bits)  # 8 QPSK symbols

    # interleave real/imag as the "symbol stream" and quantize
    sym_interleaved = np.empty(2 * len(symbols))
    sym_interleaved[0::2] = symbols.real
    sym_interleaved[1::2] = symbols.imag
    sym_q15 = to_q15(sym_interleaved)  # interleaved I,Q, length 16

    # upsample each of I and Q by sps (zero insertion), filter, matched filter
    def upsample_q15(stream_q15, sps):
        up = np.zeros(len(stream_q15) * sps, dtype=np.int32)
        up[::sps] = stream_q15
        return up

    I_q15 = sym_q15[0::2]
    Q_q15 = sym_q15[1::2]

    I_up = upsample_q15(I_q15, sps)
    Q_up = upsample_q15(Q_q15, sps)

    I_tx = fixed_point_fir(I_up, h_q15)
    Q_tx = fixed_point_fir(Q_up, h_q15)

    # NO NOISE -- deterministic bit-exact check only (see module docstring)

    I_mf = fixed_point_fir(I_tx, h_q15)
    Q_mf = fixed_point_fir(Q_tx, h_q15)

    filt_delay = span * sps // 2
    start = 2 * filt_delay
    n_symbols = len(I_q15)
    I_sampled = I_mf[start: start + n_symbols * sps: sps]
    Q_sampled = Q_mf[start: start + n_symbols * sps: sps]

    rx_bits = np.empty(2 * n_symbols, dtype=int)
    rx_bits[0::2] = (I_sampled < 0).astype(int)
    rx_bits[1::2] = (Q_sampled < 0).astype(int)

    match = np.array_equal(rx_bits, test_bits)
    print(f"Python fixed-point chain: recovered bits match input bits: {match}")
    print(f"  input:     {test_bits.tolist()}")
    print(f"  recovered: {rx_bits.tolist()}")

    # ---- write files for the C++ port to consume and for comparison ----
    np.savetxt('rrc_coeffs_q15.txt', h_q15, fmt='%d')
    np.savetxt('tx_symbols_q15.txt', sym_q15, fmt='%d')  # interleaved I,Q input
    np.savetxt('python_fixed_point_sampled.txt',
               np.column_stack([I_sampled, Q_sampled]), fmt='%d')
    np.savetxt('python_fixed_point_bits.txt', rx_bits, fmt='%d')

    print("\nWrote: rrc_coeffs_q15.txt, tx_symbols_q15.txt, "
          "python_fixed_point_sampled.txt, python_fixed_point_bits.txt")

if __name__ == "__main__":
    main()
