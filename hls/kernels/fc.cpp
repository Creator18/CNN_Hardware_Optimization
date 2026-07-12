// ============================================================================
// fc.cpp — fully-connected / matrix-vector layer, streaming datapath (INT8)
//
// Synthesized for the FC op (was assigned by analogy before). out[o] =
// sum_i in[i]*w[o][i]. The input-feature reduction is unrolled (IC_UNROLL).
// Large weight matrices are the FPGA's weak spot — this measures it.
// ============================================================================
#include "common.h"

#ifndef IC_UNROLL
#define IC_UNROLL 4
#endif

template <int IN, int OUT>
void fc(const data_t in[IN], const data_t weight[OUT][IN], acc_t out[OUT]) {
#pragma HLS INLINE off
    static data_t in_buf[IN];
#pragma HLS ARRAY_PARTITION variable=in_buf cyclic factor=IC_UNROLL dim=1
LOAD_IN:
    for (int i = 0; i < IN; i++) {
#pragma HLS PIPELINE II=1
        in_buf[i] = in[i];
    }
COMPUTE:
    for (int o = 0; o < OUT; o++) {
#pragma HLS PIPELINE II=1
        acc_t acc = 0;
        for (int i = 0; i < IN; i++) {
#pragma HLS UNROLL factor=IC_UNROLL
            acc += (acc_t)in_buf[i] * (acc_t)weight[o][i];
        }
        out[o] = acc;
    }
}

// ~ ResNet-18 classifier: 512 -> 1000.
void fc_top(const data_t in[512], const data_t weight[1000][512], acc_t out[1000]) {
#pragma HLS INTERFACE m_axi port=in     bundle=gmem0 depth=512
#pragma HLS INTERFACE m_axi port=weight bundle=gmem1 depth=512000
#pragma HLS INTERFACE m_axi port=out    bundle=gmem2 depth=1000
#pragma HLS INTERFACE s_axilite port=return
    fc<512, 1000>(in, weight, out);
}
