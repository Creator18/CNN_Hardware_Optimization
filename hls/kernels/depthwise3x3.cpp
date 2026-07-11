// ============================================================================
// depthwise3x3.cpp — depthwise 3x3 convolution, streaming datapath (INT8)
//
// The paper's central claim is that depthwise/memory-bound ops are the FPGA's
// strength. This kernel lets us measure that: one 3x3 filter per channel, no
// cross-channel accumulation, so arithmetic intensity is low and the datapath
// streams cleanly. Compare its synthesized II/DSP against conv3x3 to ground the
// depthwise-vs-conv efficiency gap that the notebook currently hard-codes
// (FPGA: 0.85 depthwise vs 0.45 conv).
// ============================================================================
#include "common.h"

template <int C, int H, int W, int K = 3>
void depthwise3x3(const data_t in[C][H][W],
                  const data_t weight[C][K][K],
                  acc_t out[C][H][W]) {
#pragma HLS INLINE off
    const int PAD = K / 2;

    for (int c = 0; c < C; c++) {
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
#pragma HLS PIPELINE II=1
                acc_t acc = 0;
                for (int ky = 0; ky < K; ky++) {
                    for (int kx = 0; kx < K; kx++) {
                        int iy = y + ky - PAD;
                        int ix = x + kx - PAD;
                        if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                            acc += (acc_t)in[c][iy][ix] * (acc_t)weight[c][ky][kx];
                        }
                    }
                }
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
