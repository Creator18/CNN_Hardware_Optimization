// ============================================================================
// pointwise.cpp — pointwise 1x1 convolution, streaming datapath (INT8)
//
// 1x1 conv is a per-pixel matrix multiply (Cin -> Cout). Common in MobileNetV2
// (expand/project). Measures the FPGA's MAC-chain efficiency without spatial
// reuse — grounds the 0.65 pointwise constant in the notebook.
// ============================================================================
#include "common.h"

template <int CIN, int COUT, int H, int W>
void pointwise(const data_t in[CIN][H][W],
               const data_t weight[COUT][CIN],
               acc_t out[COUT][H][W]) {
#pragma HLS INLINE off
    for (int oc = 0; oc < COUT; oc++) {
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
#pragma HLS PIPELINE II=1
                acc_t acc = 0;
                for (int ic = 0; ic < CIN; ic++) {
                    acc += (acc_t)in[ic][y][x] * (acc_t)weight[oc][ic];
                }
                out[oc][y][x] = acc;
            }
        }
    }
}

// ~ MobileNetV2 expand: 24 -> 144 channels at 56x56.
void pointwise_top(const data_t in[24][56][56],
                   const data_t weight[144][24],
                   acc_t out[144][56][56]) {
#pragma HLS INTERFACE m_axi port=in     bundle=gmem0 depth=75264
#pragma HLS INTERFACE m_axi port=weight bundle=gmem1 depth=3456
#pragma HLS INTERFACE m_axi port=out    bundle=gmem2 depth=451584
#pragma HLS INTERFACE s_axilite port=return
    pointwise<24, 144, 56, 56>(in, weight, out);
}
