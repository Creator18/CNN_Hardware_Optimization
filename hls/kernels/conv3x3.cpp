// ============================================================================
// conv3x3.cpp — standard 3x3 convolution, streaming datapath (INT8)
//
// Representative of the compute-heavy layers the paper assigns to the GPU.
// We synthesize it on the FPGA to measure how expensive it really is there
// (latency + II + DSP usage) rather than assuming an efficiency constant.
//
// DATAPATH (tuned): inputs and weights are burst-loaded once into on-chip BRAM.
// The input buffer is ZERO-PADDED so the compute loop has no boundary branches
// in its critical path (the naive per-tap `if` bounds check both made it
// memory-port bound and choked the scheduler). A moderate channel unroll issues
// several MACs per cycle — a realistic, well-formed design point, not a worst
// case and not a maximal one. Sweep IC_UNROLL to trace latency-vs-DSP.
// ============================================================================
#include "common.h"

// Parallel MACs across input channels (bounds DSP usage). Overridable via
// -DIC_UNROLL=N (used by the run_synth sweep) so we can trace latency vs DSP.
#ifndef IC_UNROLL
#define IC_UNROLL 4
#endif

template <int CIN, int COUT, int H, int W, int K = 3>
void conv3x3(const data_t in[CIN][H][W],
             const data_t weight[COUT][CIN][K][K],
             acc_t out[COUT][H][W]) {
#pragma HLS INLINE off
    const int PAD = K / 2;
    const int HP = H + 2 * PAD;      // padded height
    const int WP = W + 2 * PAD;      // padded width

    // Zero-padded on-chip input cache + weight cache (mapped to BRAM). Partition
    // the channel dim so IC_UNROLL consecutive channels are read in parallel.
    static data_t in_buf[CIN][HP][WP];
    static data_t w_buf[COUT][CIN][K][K];
#pragma HLS ARRAY_PARTITION variable=in_buf cyclic factor=IC_UNROLL dim=1
#pragma HLS ARRAY_PARTITION variable=w_buf  cyclic factor=IC_UNROLL dim=2

    // ---- zero the padding border, then burst-load the interior ----
INIT_PAD:
    for (int c = 0; c < CIN; c++)
        for (int y = 0; y < HP; y++)
            for (int x = 0; x < WP; x++) {
#pragma HLS PIPELINE II=1
                in_buf[c][y][x] = 0;
            }
LOAD_IN:
    for (int c = 0; c < CIN; c++)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
#pragma HLS PIPELINE II=1
                in_buf[c][y + PAD][x + PAD] = in[c][y][x];
            }
LOAD_W:
    for (int oc = 0; oc < COUT; oc++)
        for (int ic = 0; ic < CIN; ic++)
            for (int ky = 0; ky < K; ky++)
                for (int kx = 0; kx < K; kx++) {
#pragma HLS PIPELINE II=1
                    w_buf[oc][ic][ky][kx] = weight[oc][ic][ky][kx];
                }

    // ---- compute from on-chip memory (no boundary branches: buffer is padded) ----
COMPUTE:
    for (int oc = 0; oc < COUT; oc++) {
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
#pragma HLS PIPELINE II=1
                acc_t acc = 0;
                for (int ic = 0; ic < CIN; ic++) {
#pragma HLS UNROLL factor=IC_UNROLL
                    for (int ky = 0; ky < K; ky++) {
                        for (int kx = 0; kx < K; kx++) {
#pragma HLS UNROLL
                            acc += (acc_t)in_buf[ic][y + ky][x + kx] *
                                   (acc_t)w_buf[oc][ic][ky][kx];
                        }
                    }
                }
                out[oc][y][x] = acc;
            }
        }
    }
}

// Concrete instantiation to synthesize (~ ResNet-18 layer1 conv).
void conv3x3_top(const data_t in[64][56][56],
                 const data_t weight[64][64][3][3],
                 acc_t out[64][56][56]) {
#pragma HLS INTERFACE m_axi port=in     bundle=gmem0 depth=200704
#pragma HLS INTERFACE m_axi port=weight bundle=gmem1 depth=36864
#pragma HLS INTERFACE m_axi port=out    bundle=gmem2 depth=200704
#pragma HLS INTERFACE s_axilite port=return
    conv3x3<64, 64, 56, 56>(in, weight, out);
}
