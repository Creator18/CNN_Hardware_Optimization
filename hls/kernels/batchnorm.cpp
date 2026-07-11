// ============================================================================
// batchnorm.cpp — inference-time batchnorm as an elementwise affine op (INT8)
//
// At inference BN is y = scale[c]*x + shift[c] (folded params). Pure elementwise,
// memory-bound, trivial compute — the canonical "streams for free on FPGA" op
// (notebook constant 0.90). Synthesizing it grounds the near-free assumption and
// gives the per-element cost we attribute to fused BN.
// ============================================================================
#include "common.h"

template <int C, int H, int W>
void batchnorm(const data_t in[C][H][W],
               const data_t scale[C],
               const data_t shift[C],
               acc_t out[C][H][W]) {
#pragma HLS INLINE off
    for (int c = 0; c < C; c++) {
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
#pragma HLS PIPELINE II=1
                out[c][y][x] = (acc_t)in[c][y][x] * (acc_t)scale[c] + (acc_t)shift[c];
            }
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
