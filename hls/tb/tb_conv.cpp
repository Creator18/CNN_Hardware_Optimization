// ============================================================================
// tb_conv.cpp - C-simulation testbench verifying conv3x3_top (tiled datapath)
// against a naive reference convolution. Used with `vitis-run --mode hls --csim`
// to confirm functional correctness before implementation (#1). Returns 0 on
// pass, 1 on mismatch.
// ============================================================================
#include "common.h"
#include <cstdio>
#include <cstdlib>

// The design under test (defined in conv_tiled.cpp).
void conv3x3_top(const data_t in[64][56][56],
                 const data_t weight[64][64][3][3],
                 acc_t out[64][56][56]);

static data_t in[64][56][56];
static data_t weight[64][64][3][3];
static acc_t  out[64][56][56];
static acc_t  ref[64][56][56];

int main() {
    const int CIN = 64, COUT = 64, H = 56, W = 56, K = 3, PAD = 1;

    // deterministic pseudo-random INT8 data
    srand(1);
    for (int c = 0; c < CIN; c++)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                in[c][y][x] = (data_t)((rand() % 255) - 127);
    for (int oc = 0; oc < COUT; oc++)
        for (int ic = 0; ic < CIN; ic++)
            for (int ky = 0; ky < K; ky++)
                for (int kx = 0; kx < K; kx++)
                    weight[oc][ic][ky][kx] = (data_t)((rand() % 255) - 127);

    // naive reference (same padding = K/2, stride 1)
    for (int oc = 0; oc < COUT; oc++)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                acc_t acc = 0;
                for (int ic = 0; ic < CIN; ic++)
                    for (int ky = 0; ky < K; ky++)
                        for (int kx = 0; kx < K; kx++) {
                            int iy = y + ky - PAD, ix = x + kx - PAD;
                            if (iy >= 0 && iy < H && ix >= 0 && ix < W)
                                acc += (acc_t)in[ic][iy][ix] * (acc_t)weight[oc][ic][ky][kx];
                        }
                ref[oc][y][x] = acc;
            }

    conv3x3_top(in, weight, out);

    long mism = 0;
    for (int oc = 0; oc < COUT; oc++)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                if (out[oc][y][x] != ref[oc][y][x]) mism++;

    if (mism == 0) {
        printf("PASS: tiled conv matches reference on all %d outputs\n", COUT * H * W);
        return 0;
    }
    printf("FAIL: %ld mismatches\n", mism);
    return 1;
}
