#!/usr/bin/env python3
"""
parse_hls_report.py -- extract the numbers we care about from Vitis HLS
C-synthesis reports and write one committed JSON per kernel into hls/reports/.

These JSONs are the ground truth that replaces the hand-set FPGA efficiency
constants in trial2.ipynb (see hls/fpga_constants_from_hls() consumer).

Run AFTER the v++ synthesis runs (see run_synth.sh / README):
    python parse_hls_report.py

It looks for work/<kernel>/hls/syn/report/<top>_csynth.xml (the top-level
machine-readable report produced by `v++ -c --mode hls`).
"""
import glob
import json
import os
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
WORK_DIR = os.path.join(HERE, "work")
OUT_DIR = os.path.join(HERE, "reports")

# Zynq-7020 (xc7z020) total resources, for utilization fractions.
XC7Z020 = {"DSP": 220, "BRAM_18K": 280, "FF": 106400, "LUT": 53200}


def _txt(node, path, cast=float, default=None):
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
    clk = _txt(perf, ".//EstimatedClockPeriod") if perf is not None else None

    lat = perf.find(".//SummaryOfOverallLatency") if perf is not None else None
    best = _txt(lat, "Best-caseLatency", int) if lat is not None else None
    worst = _txt(lat, "Worst-caseLatency", int) if lat is not None else None
    ii = _txt(lat, "Interval-min", int) if lat is not None else None

    def res(tag):
        return _txt(area, tag, int, 0) if area is not None else 0

    dsp, bram, ff, lut = res("DSP"), res("BRAM_18K"), res("FF"), res("LUT")
    cycles = worst if worst is not None else best

    return {
        "clock_period_ns": clk,
        "latency_cycles_best": best,
        "latency_cycles_worst": worst,
        "latency_us_worst": (cycles * clk / 1000.0) if (cycles and clk) else None,
        "initiation_interval": ii,
        "dsp": dsp,
        "bram_18k": bram,
        "ff": ff,
        "lut": lut,
        "util": {
            "dsp": round(dsp / XC7Z020["DSP"], 4),
            "bram": round(bram / XC7Z020["BRAM_18K"], 4),
            "ff": round(ff / XC7Z020["FF"], 4),
            "lut": round(lut / XC7Z020["LUT"], 4),
        },
        "source_report": os.path.relpath(xml_path, HERE),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    # top-level report is <top>_csynth.xml (top functions are named *_top)
    reports = glob.glob(os.path.join(WORK_DIR, "*", "hls", "syn", "report", "*_top_csynth.xml"))
    if not reports:
        print(f"No *_top_csynth.xml found under {WORK_DIR}.")
        print("Run the v++ synthesis first (see run_synth.sh / README).")
        return

    for xml_path in reports:
        kernel = os.path.relpath(xml_path, WORK_DIR).split(os.sep)[0]
        data = parse_csynth_xml(xml_path)
        data["kernel"] = kernel
        data["part"] = "xc7z020-clg400-1"
        out = os.path.join(OUT_DIR, f"{kernel}.json")
        with open(out, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  {kernel:14s} II={data['initiation_interval']}  "
              f"lat={data['latency_us_worst']} us  DSP={data['dsp']} "
              f"({data['util']['dsp']*100:.0f}%) -> {os.path.relpath(out, HERE)}")


if __name__ == "__main__":
    main()
