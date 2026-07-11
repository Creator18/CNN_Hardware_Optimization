#!/usr/bin/env python3
"""
loo_cv.py -- leave-one-out cross-validation for the unit->ms calibration.

THE POINT
  trial2 currently calibrates with ~1 free factor per (device, model) data point,
  so cross-validation error is near zero by construction -- not a real test.
  This harness instead asks: if we calibrate on N-1 benchmark points and PREDICT
  the held-out one, how large is the error? That is a genuine generalization
  claim for "calibrated against published benchmarks".

MODEL
  For each device_class, assume:  latency_ms ~= k * emulated_units
  Calibrate k on the training fold (mean of per-point ratios, robust option:
  median), predict the held-out point, record % error. Report mean/median/max.

WIRING
  `emulated_units[(model, device_class)]` must come from the notebook's emulator
  (sum of get_block_*_cost over fused blocks). Export it once from trial2, e.g.:
      import json; json.dump(units_dict, open('calibration/emulated_units.json','w'))
  with keys like "ResNet-18|FPGA". This script loads that file if present;
  otherwise it runs a self-check on synthetic units so the harness is testable
  now, before the toolchain/notebook wiring is done.
"""
import csv
import json
import os
import statistics as stats

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "benchmarks.csv")
UNITS = os.path.join(HERE, "emulated_units.json")


def load_rows(exclude_estimates=True):
    rows = []
    with open(CSV, newline="") as f:
        for r in csv.DictReader(f):
            if exclude_estimates and r.get("is_estimate") == "1":
                continue
            r["latency_ms"] = float(r["latency_ms"])
            rows.append(r)
    return rows


def load_units():
    if os.path.exists(UNITS):
        return json.load(open(UNITS))
    return None


def loo(rows, units, estimator=stats.median):
    """Leave-one-out over rows within each device_class."""
    by_class = {}
    for r in rows:
        by_class.setdefault(r["device_class"], []).append(r)

    results = {}
    for dev_class, group in by_class.items():
        if len(group) < 2:
            print(f"  [{dev_class}] only {len(group)} point(s) -- need >=2 for LOO. "
                  f"Add more rows to benchmarks.csv.")
            continue
        errs = []
        for i, held in enumerate(group):
            train = group[:i] + group[i + 1:]
            # calibrate k = ms / units on the training fold
            ks = [t["latency_ms"] / units[f"{t['model']}|{dev_class}"] for t in train]
            k = estimator(ks)
            pred = k * units[f"{held['model']}|{dev_class}"]
            err = abs(pred - held["latency_ms"]) / held["latency_ms"] * 100
            errs.append(err)
            print(f"  [{dev_class}] held={held['model']:<12} {held['source']:<22} "
                  f"pred={pred:7.1f}ms  actual={held['latency_ms']:7.1f}ms  err={err:5.1f}%")
        results[dev_class] = {
            "mean_err_pct": round(stats.mean(errs), 2),
            "median_err_pct": round(stats.median(errs), 2),
            "max_err_pct": round(max(errs), 2),
            "n": len(group),
        }
    return results


def _synthetic_selfcheck():
    """Runs before real wiring: fabricate units so the harness is exercisable."""
    print("emulated_units.json not found -- running SYNTHETIC self-check.\n"
          "(Export real units from trial2 to get real numbers.)\n")
    rows = load_rows(exclude_estimates=True)
    units = {}
    for r in rows:
        # fake but monotonic: bigger latency -> more units, +/- noise per source
        units[f"{r['model']}|{r['device_class']}"] = r["latency_ms"] * 1000.0
    return rows, units


def main():
    units = load_units()
    if units is None:
        rows, units = _synthetic_selfcheck()
    else:
        rows = load_rows(exclude_estimates=True)
    res = loo(rows, units)
    print("\nLOO calibration error by device class:")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
