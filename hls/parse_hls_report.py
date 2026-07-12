#!/usr/bin/env python3
"""
parse_hls_report.py -- extract latency / II / resource from Vitis HLS
C-synthesis reports into committed JSON under hls/reports/.

Produces:
  reports/<op>.json         one per op (conv3x3, depthwise3x3, pointwise,
                            batchnorm, relu, pool, fc)
  reports/conv_sweep.json   conv latency-vs-DSP across IC_UNROLL (the A3 sweep)

Canonical conv uses the IC_UNROLL=4 point from the sweep. Run after the batch
synthesis (see run_synth.bat / the sweep configs). Reports live at
work/<dir>/hls/syn/report/<top>_csynth.xml.
"""
import glob
import json
import os
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
WORK_DIR = os.path.join(HERE, "work")
OUT_DIR = os.path.join(HERE, "reports")

XC7Z020 = {"DSP": 220, "BRAM_18K": 280, "FF": 106400, "LUT": 53200}

# op -> candidate work_dir names (design points). We pick the one that FITS the
# xc7z020 with the best throughput (lowest latency); if none fit, the lowest-DSP
# candidate is kept and flagged. Memory-bound ops fit best at low unroll.
OP_CANDIDATES = {
    # conv now uses the output-stationary DSP-tiled kernel (conv_tiled.cpp),
    # which removed the naive kernel's DSP cliff. Best-fitting point is chosen.
    "conv3x3":      ["conv_tiled_16x16", "conv_tiled_16x12", "conv_tiled_16x8", "conv_tiled_8x8"],
    "depthwise3x3": ["dw_u1", "dw_u2", "depthwise3x3"],
    "pointwise":    ["pointwise"],
    "batchnorm":    ["batchnorm"],
    "relu":         ["relu"],
    "pool":         ["pool"],
    "fc":           ["fc_u1", "fc"],
}
# Tiled-conv latency-vs-DSP sweep (parallel MACs = TM*TN); replaces the naive
# IC_UNROLL sweep. Keyed by TM*TN for the figure label.
CONV_SWEEP = {64: "conv_tiled_8x8", 128: "conv_tiled_16x8",
              192: "conv_tiled_16x12", 256: "conv_tiled_16x16"}


def _txt(node, path, cast=float, default=None):
    if node is None:
        return default
    el = node.find(path)
    if el is None or el.text is None:
        return default
    try:
        return cast(el.text.strip())
    except ValueError:
        return default


def parse_csynth_xml(xml_path: str) -> dict:
    root = ET.parse(xml_path).getroot()
    perf = root.find(".//PerformanceEstimates")
    area = root.find(".//AreaEstimates/Resources")
    clk = _txt(perf, ".//EstimatedClockPeriod")
    lat = perf.find(".//SummaryOfOverallLatency") if perf is not None else None
    best = _txt(lat, "Best-caseLatency", int)
    worst = _txt(lat, "Worst-caseLatency", int)
    ii = _txt(lat, "Interval-min", int)
    res = lambda t: _txt(area, t, int, 0)
    dsp, bram, ff, lut = res("DSP"), res("BRAM_18K"), res("FF"), res("LUT")
    cycles = worst if worst is not None else best
    return {
        "clock_period_ns": clk,
        "latency_cycles_best": best,
        "latency_cycles_worst": worst,
        "latency_us_worst": (cycles * clk / 1000.0) if (cycles and clk) else None,
        "initiation_interval": ii,
        "dsp": dsp, "bram_18k": bram, "ff": ff, "lut": lut,
        "util": {"dsp": round(dsp / XC7Z020["DSP"], 4),
                 "bram": round(bram / XC7Z020["BRAM_18K"], 4),
                 "ff": round(ff / XC7Z020["FF"], 4),
                 "lut": round(lut / XC7Z020["LUT"], 4)},
        "fits_xc7z020": (dsp <= XC7Z020["DSP"] and lut <= XC7Z020["LUT"]
                         and bram <= XC7Z020["BRAM_18K"] and ff <= XC7Z020["FF"]),
        "source_report": os.path.relpath(xml_path, HERE),
    }


def top_report(workdir: str):
    hits = glob.glob(os.path.join(WORK_DIR, workdir, "hls", "syn", "report", "*_top_csynth.xml"))
    return hits[0] if hits else None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    def latency(d):
        return d["latency_cycles_worst"] or d["latency_cycles_best"] or 1 << 62

    print("Per-op reports (selected design point per op):")
    for op, cands in OP_CANDIDATES.items():
        parsed = []
        for wd in cands:
            rep = top_report(wd)
            if rep:
                d = parse_csynth_xml(rep)
                d["design_point"] = wd
                parsed.append(d)
        if not parsed:
            print(f"  [skip] {op}: no reports among {cands}")
            continue
        fitting = [d for d in parsed if d["fits_xc7z020"]]
        # prefer fitting w/ best throughput (lowest latency); else lowest-DSP overall
        chosen = min(fitting, key=latency) if fitting else min(parsed, key=lambda d: d["dsp"])
        chosen["kernel"] = op
        chosen["part"] = "xc7z020-clg400-1"
        json.dump(chosen, open(os.path.join(OUT_DIR, f"{op}.json"), "w"), indent=2)
        print(f"  {op:14s} [{chosen['design_point']:<13}] II={chosen['initiation_interval']}  "
              f"DSP={chosen['dsp']} ({chosen['util']['dsp']*100:.0f}%)  "
              f"LUT={chosen['util']['lut']*100:.0f}%  fits={chosen['fits_xc7z020']}")

    print("\nConv IC_UNROLL sweep:")
    sweep = []
    for u, wd in CONV_SWEEP.items():
        rep = top_report(wd)
        if not rep:
            print(f"  [skip] unroll={u}: no report under work/{wd}")
            continue
        d = parse_csynth_xml(rep)
        cyc = d["latency_cycles_worst"] or d["latency_cycles_best"]
        sweep.append({"unroll": u, "latency_cycles": cyc,
                      "latency_us": d["latency_us_worst"], "dsp": d["dsp"],
                      "lut": d["lut"], "bram_18k": d["bram_18k"],
                      "fits_xc7z020": d["fits_xc7z020"]})
        print(f"  unroll={u:<2}  cyc={cyc:>12,}  DSP={d['dsp']:>4} "
              f"({d['util']['dsp']*100:>3.0f}%)  LUT={d['util']['lut']*100:>3.0f}%  "
              f"fits={d['fits_xc7z020']}")
    sweep.sort(key=lambda r: r["unroll"])
    json.dump(sweep, open(os.path.join(OUT_DIR, "conv_sweep.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
