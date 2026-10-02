"""
LDPC coding over BPSK/AWGN, with a from-scratch sum-product
(belief propagation) decoder. Standalone on BPSK/AWGN for now (clean
validation); can be folded into the OFDM chain afterward.

Key correctness point (easy to get wrong): coding costs bandwidth -- to
send K information bits you must transmit N > K coded bits. So at a
FIXED Eb/N0 (energy per INFORMATION bit), the energy per CODED SYMBOL is
lower by the code rate R = K/N:
    Es/N0 = R * (Eb/N0)
Get this backwards and your "coding gain" plot is fiction.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

def q_function(x):
    return 0.5 * erfc(x / np.sqrt(2))

def ber_awgn_theory(ebn0_linear):
    return q_function(np.sqrt(2 * ebn0_linear))

# ---------- build a systematic sparse LDPC code ----------

def build_ldpc(K, M, col_weight=3, seed=None):
    """
    N = K + M. Construct H = [P | I_M] (M x N), where P is M x K with
    exactly `col_weight` ones per column (random rows, no repeats) --
    this keeps H sparse (low density), which is what makes belief
    propagation tractable and is the defining property of an LDPC code.

    Systematic generator: G = [I_K | P^T] satisfies H @ G^T = 0 (mod 2)
    by construction (P + P = 0 mod 2), so encoding is just:
        codeword = [message | (message @ P^T) mod 2]
    with no Gaussian elimination needed.
    """
    rng = np.random.default_rng(seed)
    P = np.zeros((M, K), dtype=np.uint8)
    for k in range(K):
        rows = rng.choice(M, size=col_weight, replace=False)
        P[rows, k] = 1
    H = np.concatenate([P, np.eye(M, dtype=np.uint8)], axis=1)  # M x N
    N = K + M
    return H, P, N

def ldpc_encode(msg_bits, P):
    parity = (msg_bits @ P.T) % 2
    return np.concatenate([msg_bits, parity]).astype(np.uint8)

# ---------- sum-product (belief propagation) decoder ----------

def build_adjacency(H):
    M, N = H.shape
    var_to_checks = [np.nonzero(H[:, v])[0] for v in range(N)]
    check_to_vars = [np.nonzero(H[c, :])[0] for c in range(M)]
    return var_to_checks, check_to_vars

def ldpc_decode(channel_llr, H, var_to_checks, check_to_vars, max_iter=20):
    M, N = H.shape
    # message arrays: msg_v2c[v][idx into var_to_checks[v]], same for c2v
    msg_v2c = {v: np.full(len(var_to_checks[v]), channel_llr[v]) for v in range(N)}
    msg_c2v = {c: np.zeros(len(check_to_vars[c])) for c in range(M)}

    for _ in range(max_iter):
        # --- check node update (tanh rule) ---
        for c in range(M):
            vs = check_to_vars[c]
            incoming = np.array([
                msg_v2c[v][np.where(var_to_checks[v] == c)[0][0]] for v in vs
            ])
            tanh_vals = np.tanh(np.clip(incoming, -20, 20) / 2)
            prod_all = np.prod(tanh_vals)
            # leave-one-out product via division (guard against exact zeros)
            with np.errstate(divide='ignore', invalid='ignore'):
                loo = np.where(tanh_vals != 0, prod_all / tanh_vals, 0.0)
            loo = np.clip(loo, -1 + 1e-12, 1 - 1e-12)
            msg_c2v[c] = 2 * np.arctanh(loo)

        # --- variable node update ---
        total_llr = channel_llr.copy()
        for v in range(N):
            cs = var_to_checks[v]
            incoming = np.array([
                msg_c2v[c][np.where(check_to_vars[c] == v)[0][0]] for c in cs
            ])
            total_llr[v] = channel_llr[v] + np.sum(incoming)
            # extrinsic message to each check = total minus that check's own contribution
            msg_v2c[v] = total_llr[v] - incoming

        # --- early stop if syndrome satisfied ---
        hard_bits = (total_llr < 0).astype(np.uint8)
        if np.all((H @ hard_bits) % 2 == 0):
            return hard_bits, True

    return hard_bits, False

# ---------- simulation ----------

def simulate_ldpc(ebn0_db, H, P, var_to_checks, check_to_vars, K, N,
                   max_iter=20, min_block_errors=50, max_blocks=5000):
    R = K / N
    ebn0_linear = 10 ** (ebn0_db / 10)
    es_n0 = R * ebn0_linear          # coded-symbol SNR, accounts for rate loss
    sigma2 = 1 / (2 * es_n0)         # BPSK, real AWGN, unit symbol energy

    bit_errors = 0
    block_errors = 0
    total_bits = 0
    blocks = 0

    while block_errors < min_block_errors and blocks < max_blocks:
        blocks += 1
        msg = np.random.randint(0, 2, K).astype(np.uint8)
        codeword = ldpc_encode(msg, P)

        x = 1 - 2 * codeword.astype(float)          # BPSK map
        noise = np.sqrt(sigma2) * np.random.randn(N)
        y = x + noise

        channel_llr = 2 * y / sigma2
        decoded, converged = ldpc_decode(channel_llr, H, var_to_checks,
                                          check_to_vars, max_iter=max_iter)
        decoded_msg = decoded[:K]

        errs = np.sum(decoded_msg != msg)
        bit_errors += errs
        total_bits += K
        if errs > 0:
            block_errors += 1

    return bit_errors / total_bits, block_errors / blocks

def main():
    K, M = 24, 24     # rate 1/2, N = 48 (small for demo speed; note this in README)
    H, P, N = build_ldpc(K, M, col_weight=3, seed=42)
    var_to_checks, check_to_vars = build_adjacency(H)
    R = K / N
    print(f"LDPC code: N={N}, K={K}, rate={R:.2f}")

    ebn0_range_db = np.arange(0, 7, 1)
    ber_coded, bler_coded = [], []

    for ebn0_db in ebn0_range_db:
        ber, bler = simulate_ldpc(ebn0_db, H, P, var_to_checks, check_to_vars, K, N)
        ber_coded.append(ber)
        bler_coded.append(bler)
        print(f"Eb/N0 = {ebn0_db} dB  coded BER = {ber:.3e}  BLER = {bler:.3e}")

    ebn0_linear = 10 ** (ebn0_range_db / 10)
    ber_uncoded = ber_awgn_theory(ebn0_linear)

    plt.figure(figsize=(7.5, 5.5))
    plt.semilogy(ebn0_range_db, ber_coded, 'o-', label=f'LDPC coded BER (rate {R:.2f})')
    plt.semilogy(ebn0_range_db, bler_coded, '^-', label='LDPC coded BLER (block error rate)')
    plt.semilogy(ebn0_range_db, ber_uncoded, 'k--', label='Uncoded BPSK BER (theory)')
    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('Error rate')
    plt.title(f'LDPC ({N},{K}) coding gain vs uncoded BPSK, BP decoder, max_iter=20')
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.savefig('ldpc_coding_gain.png', dpi=150)
    plt.show()
    print("\nSaved plot to ldpc_coding_gain.png")

if __name__ == "__main__":
    main()
