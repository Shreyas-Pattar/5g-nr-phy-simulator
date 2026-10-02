%%writefile fixed_point_chain.cpp
// Fixed-point (Q1.15) C++ port of the RRC pulse-shaping + matched-filter
// chain. Reads the SAME quantized filter coefficients and input symbols
// the Python golden model used, runs identical integer arithmetic, and
// writes its own output for a bit-exact diff against Python's output.
//
// No noise, by design: random number generators aren't portable across
// languages, so this validates arithmetic correctness (quantization,
// rounding, saturation) on a deterministic signal, not a BER curve.

#include <cstdint>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

constexpr int32_t Q15_MAX = 32767;
constexpr int32_t Q15_MIN = -32768;

int32_t q15_saturate(int64_t x) {
    if (x > Q15_MAX) return Q15_MAX;
    if (x < Q15_MIN) return Q15_MIN;
    return static_cast<int32_t>(x);
}

std::vector<int32_t> read_ints(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        std::cerr << "Failed to open " << path << "\n";
        std::exit(1);
    }
    std::vector<int32_t> v;
    int32_t x;
    while (f >> x) v.push_back(x);
    return v;
}

void write_ints(const std::string& path, const std::vector<int32_t>& v) {
    std::ofstream f(path);
    for (int32_t x : v) f << x << "\n";
}

// Full convolution, Q1.15 in, Q1.15 out. Accumulate in int64 across ALL
// taps for one output sample, THEN round/shift/saturate ONCE -- must
// match the Python side's fixed_point_fir exactly, tap for tap.
std::vector<int32_t> fixed_point_fir(const std::vector<int32_t>& x,
                                      const std::vector<int32_t>& h) {
    int N = static_cast<int>(x.size());
    int M = static_cast<int>(h.size());
    std::vector<int32_t> y(N + M - 1, 0);

    for (int n = 0; n < N + M - 1; ++n) {
        int64_t acc = 0;
        for (int k = 0; k < M; ++k) {
            int xi = n - k;
            if (xi >= 0 && xi < N) {
                acc += static_cast<int64_t>(x[xi]) * static_cast<int64_t>(h[k]);
            }
        }
        // SAME rounding convention as Python: (acc + (1<<14)) >> 15,
        // arithmetic (floor-based) right shift on a signed 64-bit value.
        // This is well-defined and matches Python's integer >> exactly
        // on every mainstream compiler for two's-complement int64_t.
        int64_t rounded = (acc + (1LL << 14)) >> 15;
        y[n] = q15_saturate(rounded);
    }
    return y;
}

std::vector<int32_t> upsample(const std::vector<int32_t>& stream, int sps) {
    std::vector<int32_t> up(stream.size() * sps, 0);
    for (size_t i = 0; i < stream.size(); ++i) {
        up[i * sps] = stream[i];
    }
    return up;
}

int main() {
    const int span = 8, sps = 8;
    const int filt_delay = span * sps / 2;

    std::vector<int32_t> h_q15 = read_ints("rrc_coeffs_q15.txt");
    std::vector<int32_t> sym_q15 = read_ints("tx_symbols_q15.txt"); // interleaved I,Q

    int n_symbols = static_cast<int>(sym_q15.size()) / 2;
    std::vector<int32_t> I_q15(n_symbols), Q_q15(n_symbols);
    for (int i = 0; i < n_symbols; ++i) {
        I_q15[i] = sym_q15[2 * i];
        Q_q15[i] = sym_q15[2 * i + 1];
    }

    std::vector<int32_t> I_up = upsample(I_q15, sps);
    std::vector<int32_t> Q_up = upsample(Q_q15, sps);

    std::vector<int32_t> I_tx = fixed_point_fir(I_up, h_q15);
    std::vector<int32_t> Q_tx = fixed_point_fir(Q_up, h_q15);

    // NO NOISE -- see file header.

    std::vector<int32_t> I_mf = fixed_point_fir(I_tx, h_q15);
    std::vector<int32_t> Q_mf = fixed_point_fir(Q_tx, h_q15);

    int start = 2 * filt_delay;
    std::vector<int32_t> I_sampled, Q_sampled;
    std::vector<int32_t> bits;
    for (int i = 0; i < n_symbols; ++i) {
        int idx = start + i * sps;
        I_sampled.push_back(I_mf[idx]);
        Q_sampled.push_back(Q_mf[idx]);
        bits.push_back(I_mf[idx] < 0 ? 1 : 0);
        bits.push_back(Q_mf[idx] < 0 ? 1 : 0);
    }

    // write interleaved I,Q sampled values (one per line: "I Q")
    std::ofstream fs("cpp_fixed_point_sampled.txt");
    for (int i = 0; i < n_symbols; ++i) {
        fs << I_sampled[i] << " " << Q_sampled[i] << "\n";
    }
    write_ints("cpp_fixed_point_bits.txt", bits);

    std::cout << "C++ fixed-point chain complete. "
              << "Wrote cpp_fixed_point_sampled.txt, cpp_fixed_point_bits.txt\n";
    std::cout << "Recovered bits: ";
    for (int32_t b : bits) std::cout << b << " ";
    std::cout << "\n";

    return 0;
}
