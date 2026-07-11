// ============================================================================
// common.h — shared types and layer parameters for the HLS cost-grounding kernels
//
// PURPOSE
//   These kernels exist to produce *synthesis reports* (latency, initiation
//   interval, DSP/BRAM/LUT utilization) for the Zynq-7020 (xc7z020) so we can
//   replace the hand-set FPGA efficiency constants in trial2.ipynb with
//   vendor-derived numbers. They are deliberately simple, representative
//   streaming datapaths — NOT optimized production accelerators. Tune the
//   pragmas in Vitis HLS and re-synthesize to explore the design space.
//
//   Precision: INT8 activations/weights with INT32 accumulators, matching the
//   INT8 quantization assumption used throughout the paper.
// ============================================================================
#ifndef COMMON_H
#define COMMON_H

#include <ap_int.h>
#include <hls_stream.h>

typedef ap_int<8>  data_t;   // INT8 activation / weight
typedef ap_int<32> acc_t;    // INT32 accumulator

// Streaming interface: one pixel-channel element per beat.
typedef hls::stream<data_t> stream_t;

// ---------------------------------------------------------------------------
// Representative layer shapes to synthesize. Pick one per synthesis run via
// the -tclargs to run_synth.tcl, or edit here. These mirror shapes that appear
// in ResNet-18 / MobileNetV2 (see build_* in the notebooks).
// ---------------------------------------------------------------------------
// Standard 3x3 conv     : e.g. layer1 of ResNet-18   (Cin=64,  Cout=64,  H=W=56)
// Depthwise 3x3         : e.g. MobileNetV2 block      (C=96,             H=W=56)
// Pointwise 1x1 conv    : e.g. MobileNetV2 expand     (Cin=24, Cout=144, H=W=56)
// Batchnorm (elementwise): e.g. after any conv        (C=64,             H=W=56)

#endif // COMMON_H
