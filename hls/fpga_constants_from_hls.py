#!/usr/bin/env python3
"""
fpga_constants_from_hls.py -- convert HLS synthesis reports (hls/reports/*.json)
into the per-layer-type FPGA "efficiency" constants used by trial2.ipynb's
FPGAEmulator.

WHAT IT REPLACES
  trial2 currently hard-codes, e.g.:
      streaming_efficiency = {CONV: 0.45, DEPTHWISE_CONV: 0.85, ...}
  Those are guesses. Here we derive an equivalent effective throughput from the
  synthesized initiation interval:

      achieved_macs_per_cycle = mac_ops / (latency_cycles)
      efficiency = achieved_macs_per_cycle / dsp48e   # per-DSP utilization

  i.e. how close the synthesized datapath gets to one MAC per DSP per cycle.
  This gives a *measured* efficiency per op type for the xc7z020, grounded in
  the vendor tool instead of hand-set.

NOTE ON HONESTY
  These reflect the specific kernel microarchitecture in hls/kernels/. Different
  unroll/partition choices give different points. Report the design point(s) you
  actually synthesized; consider sweeping pragmas to show the range.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(HERE, "reports")

# MAC counts for the concrete shapes instantiated in hls/kernels/*_top.
# Keep these in sync with the *_top() shapes if you change them.
KERNEL_MACS = {
    "conv3x3":      64 * 64 * 56 * 56 * 3 * 3,     # Cout*Cin*H*W*K*K
    "depthwise3x3": 96 * 56 * 56 * 3 * 3,          # C*H*W*K*K
    "pointwise":    144 * 24 * 56 * 56,            # Cout*Cin*H*W
    "batchnorm":    64 * 56 * 56,                  # elementwise (1 "MAC"/elem)
}

KERNEL_TO_LAYERTYPE = {
    "conv3x3":      "CONV",
    "depthwise3x3": "DEPTHWISE_CONV",
    "pointwise":    "POINTWISE_CONV",
    "batchnorm":    "BATCHNORM",
}


def derive_constants() -> dict:
    out = {}
    for kernel, macs in KERNEL_MACS.items():
        path = os.path.join(REPORTS, f"{kernel}.json")
        if not os.path.exists(path):
            print(f"  [skip] {kernel}: no report yet ({path})")
            continue
        r = json.load(open(path))
        cycles = r.get("latency_cycles_worst") or r.get("latency_cycles_best")
        dsp = r.get("dsp48e") or 0
        if not cycles:
            print(f"  [skip] {kernel}: no latency in report")
            continue
        macs_per_cycle = macs / cycles
        eff = (macs_per_cycle / dsp) if dsp else macs_per_cycle
        lt = KERNEL_TO_LAYERTYPE[kernel]
        out[lt] = {
            "macs_per_cycle": round(macs_per_cycle, 4),
            "dsp48e": dsp,
            "efficiency_per_dsp": round(eff, 4),
            "latency_us": r.get("latency_us_worst"),
            "initiation_interval": r.get("initiation_interval"),
        }
        print(f"  {lt:16s} macs/cyc={macs_per_cycle:8.2f}  DSP={dsp:4d}  "
              f"eff/DSP={eff:.3f}")
    return out


if __name__ == "__main__":
    consts = derive_constants()
    out = os.path.join(HERE, "reports", "_fpga_constants.json")
    json.dump(consts, open(out, "w"), indent=2)
    print(f"\nWrote {os.path.relpath(out, HERE)} -- import this into trial2 to "
          f"replace hand-set streaming_efficiency.")
