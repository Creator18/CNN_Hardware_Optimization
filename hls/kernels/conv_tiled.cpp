// ============================================================================
// conv_tiled.cpp - output-stationary DSP-tiled 3x3 convolution (INT8)
//
// Replacement for the naive conv3x3, which had a DSP cliff (fit only at
// unroll=1; unroll>=2 blew past 322 DSPs). Here parallelism is split across two
// independent tile dimensions:
//   TM = output-channel tile (parallel accumulators)
//   TN = input-channel  tile (parallel MAC lanes per accumulator)
// so the DSP count scales as ~TM*TN and can be tuned to fill the 220-DSP budget
// without the LUT explosion. Inputs/weights are cached on-chip and the input is
// zero-padded (no boundary branches in the pipeline).
//
// Sweep TM/TN via -DTM=.. -DTN=.. to trace the latency-vs-DSP curve and pick a
// device-fitting point.
// ============================================================================
#include "common.h"

#ifndef TM
#define TM 8      // output channels computed in parallel
#endif
#ifndef TN
#define TN 8      // input channels reduced in parallel
#endif

template <int CIN, int COUT, int H, int W, int K = 3>
void conv_tiled(const data_t in[CIN][H][W],
                const data_t weight[COUT][CIN][K][K],
                acc_t out[COUT][H][W]) {
#pragma HLS INLINE off
    const int PAD = K / 2;
    const int HP = H + 2 * PAD, WP = W + 2 * PAD;

    static data_t in_buf[CIN][HP][WP];
    static data_t w_buf[COUT][CIN][K][K];
#pragma HLS ARRAY_PARTITION variable=in_buf cyclic factor=TN dim=1
#pragma HLS ARRAY_PARTITION variable=w_buf  cyclic factor=TM dim=1
#pragma HLS ARRAY_PARTITION variable=w_buf  cyclic factor=TN dim=2

    // ---- zero-pad, then burst-load input and weights on-chip ----
INIT:
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

    // ---- output-stationary compute ----
    // Parallelism is EXACTLY TM*TN: only the m/n tiles are unrolled. The
    // reduction over (input-channel tiles x K x K) is a single PIPELINED loop,
    // so HLS does NOT unroll it (pipelining the spatial loop instead would
    // fully unroll the whole reduction -> thousands of DSPs). int8*int8 keeps
    // each MAC on a single DSP.
    const int RED = (CIN / TN) * K * K;
COMPUTE:
    for (int oc0 = 0; oc0 < COUT; oc0 += TM) {
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                acc_t acc[TM];
#pragma HLS ARRAY_PARTITION variable=acc complete
                for (int m = 0; m < TM; m++) {
#pragma HLS UNROLL
                    acc[m] = 0;
                }
            REDUCE:
                for (int r = 0; r < RED; r++) {
#pragma HLS PIPELINE II=1
                    int ic0 = (r / (K * K)) * TN;
                    int ky = (r / K) % K;
                    int kx = r % K;
                    for (int m = 0; m < TM; m++) {
#pragma HLS UNROLL
                        acc_t psum = 0;
                        for (int n = 0; n < TN; n++) {
#pragma HLS UNROLL
                            psum += (acc_t)(in_buf[ic0 + n][y + ky][x + kx] *
                                            w_buf[oc0 + m][ic0 + n][ky][kx]);
                        }
                        acc[m] += psum;
                    }
                }
                for (int m = 0; m < TM; m++) {
#pragma HLS UNROLL
                    out[oc0 + m][y][x] = acc[m];
                }
            }
        }
    }
}

// Same shape/signature as conv3x3_top (~ ResNet-18 layer1 conv), so this is a
// drop-in replacement in the cost-grounding pipeline.
void conv3x3_top(const data_t in[64][56][56],
                 const data_t weight[64][64][3][3],
                 acc_t out[64][56][56]) {
#pragma HLS INTERFACE m_axi port=in     bundle=gmem0 depth=200704
#pragma HLS INTERFACE m_axi port=weight bundle=gmem1 depth=36864
#pragma HLS INTERFACE m_axi port=out    bundle=gmem2 depth=200704
#pragma HLS INTERFACE s_axilite port=return
    conv_tiled<64, 64, 56, 56>(in, weight, out);
}
