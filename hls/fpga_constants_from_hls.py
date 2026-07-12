#!/usr/bin/env python3
"""
fpga_constants_from_hls.py -- convert HLS synthesis reports (hls/reports/*.json)
into the FPGA cost constants consumed by trial3.ipynb.

Emits reports/_fpga_constants.json with, per layer type:
  efficiency         macs_per_cycle / DSP_SLICES(220) -- fraction of the device's
                     220-MAC/cycle peak actually achieved. SAME scale as trial2's
                     hand-set streaming_efficiency, so the swap is apples-to-apples.
  efficiency_per_dsp macs_per_cycle / DSP_used (datapath quality; supporting)
  bytes_per_cycle    per-op achieved memory throughput
and globally:
  fpga_bytes_per_cycle  effective FPGA memory bandwidth from the memory-bound
                        probes (relu/batchnorm/pool) -- grounds the roofline (A4).

Each op uses a design point that FITS the xc7z020. Memory-bound ops (depthwise,
bn, relu, pool) are efficient at LOW unroll -- unrolling them wastes DSPs without
raising throughput -- so their fitting point is chosen accordingly. Conv uses the
largest-fitting sweep point; see reports/conv_sweep.json for the full curve.
"""
import json
import os
import statistics as stats

DSP_SLICES = 220

HERE = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(HERE, "reports")

# MAC counts for the concrete top() shapes (keep in sync with kernels/*_top).
KERNEL_MACS = {
    "conv3x3":      64 * 64 * 56 * 56 * 3 * 3,
    "depthwise3x3": 96 * 56 * 56 * 3 * 3,
    "pointwise":    144 * 24 * 56 * 56,
    "batchnorm":    2 * 64 * 56 * 56,
    "relu":         64 * 56 * 56,
    "pool":         3 * 3 * 64 * 56 * 56,
    "fc":           512 * 1000,
}

# Real bytes moved through m_axi (INT8 in/params, INT32 out) -- for A4.
KERNEL_BYTES = {
    "conv3x3":      64*56*56 + 64*64*9 + 64*56*56*4,
    "depthwise3x3": 96*56*56 + 96*9 + 96*56*56*4,
    "pointwise":    24*56*56 + 144*24 + 144*56*56*4,
    "batchnorm":    64*56*56 + 128 + 64*56*56*4,
    "relu":         64*56*56 + 64*56*56*4,
    "pool":         64*112*112 + 64*56*56*4,
    "fc":           512 + 512000 + 1000*4,
}

KERNEL_TO_LAYERTYPE = {
    "conv3x3": "CONV", "depthwise3x3": "DEPTHWISE_CONV", "pointwise": "POINTWISE_CONV",
    "batchnorm": "BATCHNORM", "relu": "RELU", "pool": "POOL", "fc": "FC",
}

# Memory-bound kernels used to estimate effective FPGA memory bandwidth.
MEM_BOUND_PROBES = ["relu", "batchnorm", "pool"]


def load(kernel):
    p = os.path.join(REPORTS, f"{kernel}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    out = {"_note": "efficiency = macs_per_cycle/220 (fraction of DSP peak, same "
                    "scale as trial2 hand-set); fpga_bytes_per_cycle grounds the "
                    "memory roofline. INT8; each op at a device-fitting point."}
    bytes_per_cycle_samples = {}

    for kernel, macs in KERNEL_MACS.items():
        r = load(kernel)
        if r is None:
            print(f"  [skip] {kernel}: no report")
            continue
        cyc = r.get("latency_cycles_worst") or r.get("latency_cycles_best")
        dsp = r.get("dsp") or 0
        if not cyc:
            print(f"  [skip] {kernel}: no latency")
            continue
        macs_per_cycle = macs / cyc
        efficiency = macs_per_cycle / DSP_SLICES          # fraction of 220-MAC/cyc peak
        eff_per_dsp = (macs_per_cycle / dsp) if dsp else macs_per_cycle
        bpc = KERNEL_BYTES[kernel] / cyc
        bytes_per_cycle_samples[kernel] = bpc
        out[KERNEL_TO_LAYERTYPE[kernel]] = {
            "efficiency": round(efficiency, 4),
            "macs_per_cycle": round(macs_per_cycle, 4),
            "dsp": dsp,
            "efficiency_per_dsp": round(eff_per_dsp, 4),
            "bytes_per_cycle": round(bpc, 3),
            "design_point": r.get("design_point"),
            "latency_us": r.get("latency_us_worst"),
            "fits_xc7z020": r.get("fits_xc7z020"),
        }
        print(f"  {KERNEL_TO_LAYERTYPE[kernel]:16s} eff={efficiency:.4f}  "
              f"macs/cyc={macs_per_cycle:7.2f}  DSP={dsp:<4} fits={r.get('fits_xc7z020')}")

    # A4: effective FPGA memory bandwidth from the memory-bound probes.
    probe_bpc = [bytes_per_cycle_samples[k] for k in MEM_BOUND_PROBES
                 if k in bytes_per_cycle_samples]
    if probe_bpc:
        out["fpga_bytes_per_cycle"] = round(stats.median(probe_bpc), 3)
        print(f"\n  fpga_bytes_per_cycle (median of {MEM_BOUND_PROBES}) = "
              f"{out['fpga_bytes_per_cycle']}")

    json.dump(out, open(os.path.join(REPORTS, "_fpga_constants.json"), "w"), indent=2)
    print("\nWrote reports/_fpga_constants.json")


if __name__ == "__main__":
    main()
