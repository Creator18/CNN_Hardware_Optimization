// ============================================================================
// relu.cpp — elementwise ReLU, streaming datapath (INT8)
//
// Synthesized (rather than assigned by analogy) so trial3 has a real number for
// the RELU op. Pure elementwise max(0,x): trivial compute, memory-bound — a
// second probe (with batchnorm) for the FPGA effective memory bandwidth (A4).
// ============================================================================
#include "common.h"

#ifndef C_UNROLL
#define C_UNROLL 4
#endif

template <int C, int H, int W>
void relu(const data_t in[C][H][W], acc_t out[C][H][W]) {
#pragma HLS INLINE off
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
#pragma HLS PIPELINE II=1
            for (int c = 0; c < C; c++) {
#pragma HLS UNROLL factor=C_UNROLL
                data_t v = in[c][y][x];
                out[c][y][x] = (v > 0) ? (acc_t)v : (acc_t)0;
            }
        }
}

// ~ 64x56x56 activation.
void relu_top(const data_t in[64][56][56], acc_t out[64][56][56]) {
#pragma HLS INTERFACE m_axi port=in  bundle=gmem0 depth=200704
#pragma HLS INTERFACE m_axi port=out bundle=gmem1 depth=200704
#pragma HLS INTERFACE s_axilite port=return
    relu<64, 56, 56>(in, out);
}
