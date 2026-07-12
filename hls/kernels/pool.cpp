// ============================================================================
// pool.cpp — max pooling, streaming datapath (INT8)
//
// Synthesized for the POOL op. Windowed max over KxK with stride S; comparator
// tree, low compute. Input cached on-chip; channel-unrolled.
// ============================================================================
#include "common.h"

#ifndef C_UNROLL
#define C_UNROLL 4
#endif

template <int C, int HIN, int WIN, int HOUT, int WOUT, int K, int S>
void maxpool(const data_t in[C][HIN][WIN], acc_t out[C][HOUT][WOUT]) {
#pragma HLS INLINE off
    static data_t in_buf[C][HIN][WIN];
#pragma HLS ARRAY_PARTITION variable=in_buf cyclic factor=C_UNROLL dim=1
LOAD_IN:
    for (int c = 0; c < C; c++)
        for (int y = 0; y < HIN; y++)
            for (int x = 0; x < WIN; x++) {
#pragma HLS PIPELINE II=1
                in_buf[c][y][x] = in[c][y][x];
            }
COMPUTE:
    for (int oy = 0; oy < HOUT; oy++)
        for (int ox = 0; ox < WOUT; ox++) {
#pragma HLS PIPELINE II=1
            for (int c = 0; c < C; c++) {
#pragma HLS UNROLL factor=C_UNROLL
                data_t m = in_buf[c][oy * S][ox * S];
                for (int ky = 0; ky < K; ky++)
                    for (int kx = 0; kx < K; kx++) {
#pragma HLS UNROLL
                        int iy = oy * S + ky, ix = ox * S + kx;
                        if (iy < HIN && ix < WIN) {
                            data_t v = in_buf[c][iy][ix];
                            if (v > m) m = v;
                        }
                    }
                out[c][oy][ox] = (acc_t)m;
            }
        }
}

// ~ ResNet-18 maxpool: 64ch, 112x112 -> 56x56, K=3, S=2.
void pool_top(const data_t in[64][112][112], acc_t out[64][56][56]) {
#pragma HLS INTERFACE m_axi port=in  bundle=gmem0 depth=802816
#pragma HLS INTERFACE m_axi port=out bundle=gmem1 depth=200704
#pragma HLS INTERFACE s_axilite port=return
    maxpool<64, 112, 112, 56, 56, 3, 2>(in, out);
}
