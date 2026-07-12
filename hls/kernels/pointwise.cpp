// ============================================================================
// pointwise.cpp — pointwise 1x1 convolution, streaming datapath (INT8)
//
// MATCHED-PARALLELISM version: inputs/weights cached on-chip; the input-channel
// reduction is unrolled (IC_UNROLL lanes) exactly like conv3x3, so its eff/DSP
// is a fair comparison instead of the earlier single-lane result.
// ============================================================================
#include "common.h"

#ifndef IC_UNROLL
#define IC_UNROLL 4
#endif

template <int CIN, int COUT, int H, int W>
void pointwise(const data_t in[CIN][H][W],
               const data_t weight[COUT][CIN],
               acc_t out[COUT][H][W]) {
#pragma HLS INLINE off
    static data_t in_buf[CIN][H][W];
    static data_t w_buf[COUT][CIN];
#pragma HLS ARRAY_PARTITION variable=in_buf cyclic factor=IC_UNROLL dim=1
#pragma HLS ARRAY_PARTITION variable=w_buf  cyclic factor=IC_UNROLL dim=2

LOAD_IN:
    for (int c = 0; c < CIN; c++)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
#pragma HLS PIPELINE II=1
                in_buf[c][y][x] = in[c][y][x];
            }
LOAD_W:
    for (int oc = 0; oc < COUT; oc++)
        for (int ic = 0; ic < CIN; ic++) {
#pragma HLS PIPELINE II=1
            w_buf[oc][ic] = weight[oc][ic];
        }

COMPUTE:
    for (int oc = 0; oc < COUT; oc++) {
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
#pragma HLS PIPELINE II=1
                acc_t acc = 0;
                for (int ic = 0; ic < CIN; ic++) {
#pragma HLS UNROLL factor=IC_UNROLL
                    acc += (acc_t)in_buf[ic][y][x] * (acc_t)w_buf[oc][ic];
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
