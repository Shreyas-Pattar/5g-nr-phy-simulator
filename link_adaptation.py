"""
Link adaptation (AMC -- Adaptive Modulation and Coding).

Reuses the LDPC machinery from step 4 (ldpc_bp_decoder.py). Three MCS
candidates, deliberately spanning robust -> aggressive:
  MCS0: QPSK + LDPC rate 1/2  (most robust, lowest throughput ceiling)
  MCS1: QPSK + LDPC rate 3/4  (higher throughput, less robust)
  MCS2: uncoded QPSK          (highest throughput ceiling, fragile)

Throughput metric (standard AMC definition, no retransmission credit):
    throughput (info bits / channel symbol) = R * m * (1 - BLER)
where R = code rate, m = bits per modulation symbol.

"Link adaptation" = at each SNR, simulate/compute every MCS's BLER and
throughput, then pick whichever MCS gives the HIGHEST throughput at that
SNR -- not just the fastest one blindly. Plotting all fixed-MCS curves
plus the adaptive choice should show the adaptive curve tracing the upper
envelope of the fixed curves.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

def q_function(x):
    return 0.5 * erfc(x / np.sqrt(2))

def build_ldpc(K, M, col_weight=3, seed=None):
    """
    N = K + M. Construct H = [P | I_M] (M x N), where P is M x K with
    exactly `col_weight` ones per column (random rows, no repeats) --
    keeps H sparse (low density), the defining LDPC property.
    Systematic generator G = [I_K | P^T] satisfies H @ G^T = 0 (mod 2).
    """
    rng = np.random.default_rng(seed)
    P = np.zeros((M, K), dtype=np.uint8)
    for k in range(K):
        rows = rng.choice(M, size=col_weight, replace=False)
        P[rows, k] = 1
    H = np.concatenate([P, np.eye(M, dtype=np.uint8)], axis=1)
    N = K + M
    return H, P, N

def ldpc_encode(msg_bits, P):
    parity = (msg_bits @ P.T) % 2
    return np.concatenate([msg_bits, parity]).astype(np.uint8)

def build_adjacency(H):
    M, N = H.shape
    var_to_checks = [np.nonzero(H[:, v])[0] for v in range(N)]
    check_to_vars = [np.nonzero(H[c, :])[0] for c in range(M)]
    return var_to_checks, check_to_vars

def ldpc_decode(channel_llr, H, var_to_checks, check_to_vars, max_iter=20):
    M, N = H.shape
    msg_v2c = {v: np.full(len(var_to_checks[v]), channel_llr[v]) for v in range(N)}
    msg_c2v = {c: np.zeros(len(check_to_vars[c])) for c in range(M)}

    for _ in range(max_iter):
        for c in range(M):
            vs = check_to_vars[c]
            incoming = np.array([
                msg_v2c[v][np.where(var_to_checks[v] == c)[0][0]] for v in vs
            ])
            tanh_vals = np.tanh(np.clip(incoming, -20, 20) / 2)
            prod_all = np.prod(tanh_vals)
            with np.errstate(divide='ignore', invalid='ignore'):
                loo = np.where(tanh_vals != 0, prod_all / tanh_vals, 0.0)
            loo = np.clip(loo, -1 + 1e-12, 1 - 1e-12)
            msg_c2v[c] = 2 * np.arctanh(loo)

        total_llr = channel_llr.copy()
        for v in range(N):
            cs = var_to_checks[v]
            incoming = np.array([
                msg_c2v[c][np.where(check_to_vars[c] == v)[0][0]] for c in cs
            ])
            total_llr[v] = channel_llr[v] + np.sum(incoming)
            msg_v2c[v] = total_llr[v] - incoming

        hard_bits = (total_llr < 0).astype(np.uint8)
        if np.all((H @ hard_bits) % 2 == 0):
            return hard_bits, True

    return hard_bits, False
    
# ---------- QPSK with Gray mapping (bit-pair -> symbol) ----------

def qpsk_mod_bits(coded_bits):
    """coded_bits length must be even. Gray mapping identical convention
    used throughout this project: b0 -> real sign, b1 -> imag sign."""
    b0 = coded_bits[0::2]
    b1 = coded_bits[1::2]
    i = 1 - 2 * b0.astype(float)
    q = 1 - 2 * b1.astype(float)
    return (i + 1j * q) / np.sqrt(2)

def qpsk_llr_demod(y, sigma2_dim):
    """Per-dimension LLR demod (valid because Gray-coded QPSK's I and Q
    are independent/orthogonal, same reasoning as the earlier minimum-
    distance QPSK demapper, just soft instead of hard)."""
    llr = np.empty(2 * len(y))
    llr[0::2] = 2 * y.real / sigma2_dim
    llr[1::2] = 2 * y.imag / sigma2_dim
    return llr

# ---------- MCS 0 / 1 : QPSK + LDPC (coded) ----------

def simulate_coded_qpsk_bler(ebn0_db, H, P, var_to_checks, check_to_vars,
                              K, N, max_iter=20, min_block_errors=50, max_blocks=4000):
    R = K / N
    m = 2  # QPSK
    ebn0_linear = 10 ** (ebn0_db / 10)
    es_n0 = R * m * ebn0_linear
    sigma2_dim = 1 / (2 * es_n0)   # noise variance per real dimension

    block_errors = 0
    blocks = 0
    while block_errors < min_block_errors and blocks < max_blocks:
        blocks += 1
        msg = np.random.randint(0, 2, K).astype(np.uint8)
        codeword = ldpc_encode(msg, P)

        symbols = qpsk_mod_bits(codeword)
        noise = np.sqrt(sigma2_dim) * (np.random.randn(*symbols.shape)
                                        + 1j * np.random.randn(*symbols.shape))
        y = symbols + noise

        channel_llr = qpsk_llr_demod(y, sigma2_dim)
        decoded, converged = ldpc_decode(channel_llr, H, var_to_checks,
                                          check_to_vars, max_iter=max_iter)
        if np.any(decoded[:K] != msg):
            block_errors += 1

    return block_errors / blocks

# ---------- MCS 2 : uncoded QPSK (exact closed form) ----------

def uncoded_qpsk_bler(ebn0_db, block_bits=48):
    """For uncoded transmission, bit errors ARE iid Bernoulli(BER) across
    the block (no decoder correlation), so BLER = 1-(1-BER)^block_bits is
    EXACT here, not an approximation."""
    ebn0_linear = 10 ** (ebn0_db / 10)
    ber = q_function(np.sqrt(2 * ebn0_linear))  # same per-bit formula as BPSK/QPSK
    return 1 - (1 - ber) ** block_bits

# ---------- throughput + adaptation ----------

def throughput(R, m, bler):
    return R * m * (1 - bler)

def main():
    # Build two LDPC codes sharing N=48 for a fair comparison
    K0, M0 = 24, 24          # rate 1/2
    K1, M1 = 36, 12          # rate 3/4
    H0, P0, N0_ = build_ldpc(K0, M0, col_weight=3, seed=42)
    H1, P1, N1_ = build_ldpc(K1, M1, col_weight=3, seed=7)
    v2c0, c2v0 = build_adjacency(H0)
    v2c1, c2v1 = build_adjacency(H1)
    R0, R1 = K0 / N0_, K1 / N1_
    m = 2  # QPSK for both coded MCS
    print(f"MCS0: QPSK + LDPC rate {R0:.2f} (N={N0_},K={K0})")
    print(f"MCS1: QPSK + LDPC rate {R1:.2f} (N={N1_},K={K1})")
    print("MCS2: uncoded QPSK (rate 1.0)")

    ebn0_range_db = np.arange(-2, 11, 1)

    bler0, bler1, bler2 = [], [], []
    tput0, tput1, tput2 = [], [], []

    for ebn0_db in ebn0_range_db:
        b0 = simulate_coded_qpsk_bler(ebn0_db, H0, P0, v2c0, c2v0, K0, N0_)
        b1 = simulate_coded_qpsk_bler(ebn0_db, H1, P1, v2c1, c2v1, K1, N1_)
        b2 = uncoded_qpsk_bler(ebn0_db, block_bits=N0_)

        bler0.append(b0); bler1.append(b1); bler2.append(b2)
        tput0.append(throughput(R0, m, b0))
        tput1.append(throughput(R1, m, b1))
        tput2.append(throughput(1.0, m, b2))

        print(f"Eb/N0={ebn0_db:3d} dB  "
              f"MCS0 BLER={b0:.3e} tput={tput0[-1]:.2f} | "
              f"MCS1 BLER={b1:.3e} tput={tput1[-1]:.2f} | "
              f"MCS2 BLER={b2:.3e} tput={tput2[-1]:.2f}")

    tput_matrix = np.vstack([tput0, tput1, tput2])
    best_idx = np.argmax(tput_matrix, axis=0)
    adaptive_tput = tput_matrix[best_idx, np.arange(len(ebn0_range_db))]
    mcs_names = ['MCS0 (QPSK, R=1/2)', 'MCS1 (QPSK, R=3/4)', 'MCS2 (uncoded QPSK)']

    print("\nAdaptive MCS choice by SNR:")
    for db, idx in zip(ebn0_range_db, best_idx):
        print(f"  {db:3d} dB -> {mcs_names[idx]}")

    plt.figure(figsize=(8, 5.5))
    plt.plot(ebn0_range_db, tput0, 'o-', label='MCS0: QPSK, LDPC R=1/2 (robust)')
    plt.plot(ebn0_range_db, tput1, 's-', label='MCS1: QPSK, LDPC R=3/4')
    plt.plot(ebn0_range_db, tput2, '^-', label='MCS2: uncoded QPSK (aggressive)')
    plt.plot(ebn0_range_db, adaptive_tput, 'k*--', markersize=12,
              label='Adaptive (picks best MCS per SNR)', linewidth=2)
    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('Throughput (info bits / channel symbol)')
    plt.title('Link adaptation: adaptive MCS traces the upper envelope')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('link_adaptation_throughput.png', dpi=150)
    plt.show()
    print("\nSaved plot to link_adaptation_throughput.png")

if __name__ == "__main__":
    main()
