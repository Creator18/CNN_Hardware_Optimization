// ============================================================================
// batchnorm.cpp — inference-time batchnorm as an elementwise affine op (INT8)
//
// MATCHED-PARALLELISM version: channel-unrolled like the others. BN is
// memory-bound (one read + one write per element, trivial compute), so even
// unrolled it stays limited by memory throughput — which makes it (and relu)
// our probe for the FPGA effective memory bandwidth (A4). y = scale*x + shift.
// ============================================================================
#include "common.h"

#ifndef C_UNROLL
#define C_UNROLL 4
#endif

template <int C, int H, int W>
void batchnorm(const data_t in[C][H][W],
               const data_t scale[C],
               const data_t shift[C],
               acc_t out[C][H][W]) {
#pragma HLS INLINE off
    static data_t s_buf[C], b_buf[C];
#pragma HLS ARRAY_PARTITION variable=s_buf cyclic factor=C_UNROLL dim=1
#pragma HLS ARRAY_PARTITION variable=b_buf cyclic factor=C_UNROLL dim=1
LOAD_P:
    for (int c = 0; c < C; c++) {
#pragma HLS PIPELINE II=1
        s_buf[c] = scale[c];
        b_buf[c] = shift[c];
    }
COMPUTE:
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
#pragma HLS PIPELINE II=1
            for (int c = 0; c < C; c++) {
#pragma HLS UNROLL factor=C_UNROLL
                out[c][y][x] = (acc_t)in[c][y][x] * (acc_t)s_buf[c] + (acc_t)b_buf[c];
            }
        }
}

// ~ post-conv BN at 64x56x56.
void batchnorm_top(const data_t in[64][56][56],
                   const data_t scale[64],
                   const data_t shift[64],
                   acc_t out[64][56][56]) {
#pragma HLS INTERFACE m_axi port=in    bundle=gmem0 depth=200704
#pragma HLS INTERFACE m_axi port=scale bundle=gmem1 depth=64
#pragma HLS INTERFACE m_axi port=shift bundle=gmem1 depth=64
#pragma HLS INTERFACE m_axi port=out   bundle=gmem2 depth=200704
#pragma HLS INTERFACE s_axilite port=return
    batchnorm<64, 56, 56>(in, scale, shift, out);
}
