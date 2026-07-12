// ============================================================================
// depthwise3x3.cpp — depthwise 3x3 convolution, streaming datapath (INT8)
//
// MATCHED-PARALLELISM version: like conv3x3, inputs/weights are cached on-chip
// and the input is zero-padded (no boundary branches). Channels are the unroll
// dimension (C_UNROLL parallel lanes) so its eff/DSP is comparable to conv's
// — the earlier single-lane version made depthwise look artificially cheap.
// ============================================================================
#include "common.h"

#ifndef C_UNROLL
#define C_UNROLL 4
#endif

template <int C, int H, int W, int K = 3>
void depthwise3x3(const data_t in[C][H][W],
                  const data_t weight[C][K][K],
                  acc_t out[C][H][W]) {
#pragma HLS INLINE off
    const int PAD = K / 2;
    const int HP = H + 2 * PAD, WP = W + 2 * PAD;

    static data_t in_buf[C][HP][WP];
    static data_t w_buf[C][K][K];
#pragma HLS ARRAY_PARTITION variable=in_buf cyclic factor=C_UNROLL dim=1
#pragma HLS ARRAY_PARTITION variable=w_buf  cyclic factor=C_UNROLL dim=1

INIT_PAD:
    for (int c = 0; c < C; c++)
        for (int y = 0; y < HP; y++)
            for (int x = 0; x < WP; x++) {
#pragma HLS PIPELINE II=1
                in_buf[c][y][x] = 0;
            }
LOAD_IN:
    for (int c = 0; c < C; c++)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
#pragma HLS PIPELINE II=1
                in_buf[c][y + PAD][x + PAD] = in[c][y][x];
            }
LOAD_W:
    for (int c = 0; c < C; c++)
        for (int ky = 0; ky < K; ky++)
            for (int kx = 0; kx < K; kx++) {
#pragma HLS PIPELINE II=1
                w_buf[c][ky][kx] = weight[c][ky][kx];
            }

COMPUTE:
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
#pragma HLS PIPELINE II=1
            for (int c = 0; c < C; c++) {
#pragma HLS UNROLL factor=C_UNROLL
                acc_t acc = 0;
                for (int ky = 0; ky < K; ky++)
                    for (int kx = 0; kx < K; kx++)
#pragma HLS UNROLL
                        acc += (acc_t)in_buf[c][y + ky][x + kx] * (acc_t)w_buf[c][ky][kx];
                out[c][y][x] = acc;
            }
        }
    }
}

// ~ MobileNetV2 depthwise stage.
void depthwise3x3_top(const data_t in[96][56][56],
                      const data_t weight[96][3][3],
                      acc_t out[96][56][56]) {
#pragma HLS INTERFACE m_axi port=in     bundle=gmem0 depth=301056
#pragma HLS INTERFACE m_axi port=weight bundle=gmem1 depth=864
#pragma HLS INTERFACE m_axi port=out    bundle=gmem2 depth=301056
#pragma HLS INTERFACE s_axilite port=return
    depthwise3x3<96, 56, 56>(in, weight, out);
}
