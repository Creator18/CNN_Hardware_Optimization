// ============================================================================
// conv3x3.cpp — standard 3x3 convolution, streaming datapath (INT8)
//
// Representative of the compute-heavy layers the paper assigns to the GPU.
// We synthesize it on the FPGA to measure how expensive it really is there
// (latency + II + DSP usage) rather than assuming an efficiency constant.
//
// Microarchitecture: single accumulation pipeline over (Cout, H, W, Cin, 3, 3).
// The II/latency reported by HLS is what feeds the FPGA cost model. Increase
// parallelism (UNROLL / ARRAY_PARTITION) to model a wider datapath and observe
// the DSP/latency tradeoff — that tradeoff is exactly the "efficiency" we want
// to ground empirically.
// ============================================================================
#include "common.h"

template <int CIN, int COUT, int H, int W, int K = 3>
void conv3x3(const data_t in[CIN][H][W],
             const data_t weight[COUT][CIN][K][K],
             acc_t out[COUT][H][W]) {
#pragma HLS INLINE off
    const int PAD = K / 2;

    for (int oc = 0; oc < COUT; oc++) {
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
#pragma HLS PIPELINE II=1
                acc_t acc = 0;
                for (int ic = 0; ic < CIN; ic++) {
                    for (int ky = 0; ky < K; ky++) {
                        for (int kx = 0; kx < K; kx++) {
                            int iy = y + ky - PAD;
                            int ix = x + kx - PAD;
                            if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                                acc += (acc_t)in[ic][iy][ix] * (acc_t)weight[oc][ic][ky][kx];
                            }
                        }
                    }
                }
                out[oc][y][x] = acc;
            }
        }
    }
}

// Concrete instantiation to synthesize. Edit the shape (or template via tcl)
// to match the layer you are grounding. This one ~ ResNet-18 layer1 conv.
void conv3x3_top(const data_t in[64][56][56],
                 const data_t weight[64][64][3][3],
                 acc_t out[64][56][56]) {
#pragma HLS INTERFACE m_axi port=in     bundle=gmem0 depth=200704
#pragma HLS INTERFACE m_axi port=weight bundle=gmem1 depth=36864
#pragma HLS INTERFACE m_axi port=out    bundle=gmem2 depth=200704
#pragma HLS INTERFACE s_axilite port=return
    conv3x3<64, 64, 56, 56>(in, weight, out);
}
