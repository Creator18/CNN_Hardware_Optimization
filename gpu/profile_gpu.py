#!/usr/bin/env python3
"""
profile_gpu.py -- ground the GPU side of the cost model with real measurements
(companion to hls/ which grounds the FPGA side).

Trial 2 hand-set a per-op GPU `parallel_efficiency` (how well each op uses the
GPU). Here we MEASURE it on the local CUDA GPU: run each op type at batch=1 on
the representative shapes used by the HLS kernels, time it with CUDA events, and
compute achieved MAC throughput. Per-op efficiency = achieved / peak (the best
op), i.e. how close each op gets to the device's realized peak -- the same 0..1
knob the emulator's parallel_efficiency represents.

Writes gpu/reports/_gpu_constants.json (per-op efficiency + latency + GMAC/s).

Note: this is measured on the local GPU (e.g. RTX 4070), so absolute times are
not Jetson times -- but the *relative* per-op utilization pattern (which ops use
the GPU well vs poorly at batch=1) is what parallel_efficiency encodes, and that
is measured here rather than guessed.
"""
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "reports")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32          # match the emulator's FP32 roofline assumption
N_WARMUP, N_RUNS = 30, 200

# (op_key, mac_ops, builder returning (module_or_fn, input_tensor)); shapes match
# the hls/kernels/*_top() instantiations for consistency across the two studies.
def _conv():   return nn.Conv2d(64, 64, 3, padding=1), torch.randn(1, 64, 56, 56)
def _dw():     return nn.Conv2d(96, 96, 3, padding=1, groups=96), torch.randn(1, 96, 56, 56)
def _pw():     return nn.Conv2d(24, 144, 1), torch.randn(1, 24, 56, 56)
def _bn():     return nn.BatchNorm2d(64).eval(), torch.randn(1, 64, 56, 56)
def _relu():   return nn.ReLU(), torch.randn(1, 64, 56, 56)
def _pool():   return nn.MaxPool2d(3, 2), torch.randn(1, 64, 112, 112)
def _fc():     return nn.Linear(512, 1000), torch.randn(1, 512)

OPS = {
    "CONV":           (9 * 64 * 64 * 56 * 56, _conv),
    "DEPTHWISE_CONV": (9 * 96 * 56 * 56,      _dw),
    "POINTWISE_CONV": (24 * 144 * 56 * 56,    _pw),
    "BATCHNORM":      (2 * 64 * 56 * 56,      _bn),
    "RELU":           (64 * 56 * 56,          _relu),
    "POOL":           (9 * 64 * 56 * 56,      _pool),
    "FC":             (512 * 1000,            _fc),
}


def time_op(mod, x) -> float:
    """Median GPU execution time in seconds (CUDA-event timed, batch=1)."""
    mod = mod.to(DEV).to(DTYPE) if isinstance(mod, nn.Module) else mod
    x = x.to(DEV).to(DTYPE)
    with torch.no_grad():
        for _ in range(N_WARMUP):
            mod(x)
        torch.cuda.synchronize()
        times = []
        for _ in range(N_RUNS):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            mod(x)
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e) / 1000.0)  # ms -> s
    times.sort()
    return times[len(times) // 2]


def main():
    if DEV.type != "cuda":
        print("CUDA not available; this profiler needs a GPU.")
        return
    os.makedirs(OUT, exist_ok=True)
    raw = {}
    for op, (macs, build) in OPS.items():
        mod, x = build()
        t = time_op(mod, x)
        raw[op] = {"mac_ops": macs, "latency_us": t * 1e6, "achieved_gmac_s": macs / t / 1e9}

    peak = max(r["achieved_gmac_s"] for r in raw.values())  # best op = realized peak
    out = {"_note": "efficiency = achieved GMAC/s / peak GMAC/s (batch=1, FP32, "
                    f"measured on {torch.cuda.get_device_name(0)}). Replaces "
                    "trial2 hand-set parallel_efficiency.",
           "device": torch.cuda.get_device_name(0), "peak_gmac_s": round(peak, 2)}
    print(f"GPU per-op profile (batch=1, FP32) on {torch.cuda.get_device_name(0)}:")
    print(f"  {'op':<16}{'lat(us)':>10}{'GMAC/s':>10}{'efficiency':>12}")
    for op, r in raw.items():
        eff = r["achieved_gmac_s"] / peak
        out[op] = {"efficiency": round(eff, 4),
                   "latency_us": round(r["latency_us"], 3),
                   "achieved_gmac_s": round(r["achieved_gmac_s"], 3)}
        print(f"  {op:<16}{r['latency_us']:>10.2f}{r['achieved_gmac_s']:>10.2f}{eff:>12.4f}")
    json.dump(out, open(os.path.join(OUT, "_gpu_constants.json"), "w"), indent=2)
    print(f"\nWrote {os.path.relpath(os.path.join(OUT, '_gpu_constants.json'), HERE)}")


if __name__ == "__main__":
    main()
