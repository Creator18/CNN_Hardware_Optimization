# Adaptive CNN Acceleration: Complexity-Aware Mapping on FPGA and GPU

Research code for [`Adaptive_CNN_Acceleration.pdf`](Adaptive_CNN_Acceleration.pdf)
(A. Basu, Northeastern University). The project studies partitioning CNN
inference across a **GPU + FPGA**, assigning each layer to the device that suits
it — compute-heavy convolutions to the GPU, memory-bound streaming ops
(depthwise conv, batchnorm, ReLU, pooling) to the FPGA — with operator fusion,
INT8 compression, and calibration-based scheduling.

## Phases

| Phase | Notebook | Result figures | What it does |
|-------|----------|----------------|--------------|
| 1 — Analytical | [`trial1.ipynb`](trial1.ipynb) | [`ResultImages_testingPhase/`](ResultImages_testingPhase) | Roofline/formula cost models across 6 architectures; partition optimizer; latency predictor. |
| 2 — Emulation | [`trial2.ipynb`](trial2.ipynb) | [`ResultImages_emulationPhase/`](ResultImages_emulationPhase) | Functional emulation with real tensor ops; reveals memory-bound behavior; calibration to published benchmarks. |
| 3 — Grounded *(in progress)* | `trial3.ipynb` *(planned)* | `ResultImages_groundedPhase/` *(planned)* | Replaces hand-set assumptions with **HLS-synthesized** FPGA costs and a **predictive** (leave-one-out) calibration. |

> Note the legacy naming: `trial1` pairs with `ResultImages_testingPhase`,
> `trial2` with `ResultImages_emulationPhase`. Phase 3 follows the same
> `trialN` / `ResultImages_*Phase` convention.

## Phase-3 supporting work (hardware-free grounding)

| Dir | Purpose |
|-----|---------|
| [`hls/`](hls) | Vitis HLS kernels + flow that synthesize conv / depthwise / pointwise / batchnorm for the **Zynq-7020 (xc7z020)** and export per-op latency, initiation interval, and DSP/BRAM/LUT into `hls/reports/*.json`. These replace the hand-set FPGA efficiency constants. See [`hls/README.md`](hls/README.md). |
| [`calibration/`](calibration) | Published-benchmark dataset + **leave-one-out cross-validation** harness, turning the unit→ms calibration from a per-point fit into a real generalization test. See [`calibration/schema.md`](calibration/schema.md). |

These need no FPGA board — HLS C-synthesis runs on CPU (free Vivado Basic tier).

## Reproducing

- **Notebooks:** open in Jupyter; phase 1/2 are self-contained (numpy / PyTorch /
  scikit-learn / matplotlib).
- **HLS grounding:** install Vitis HLS 2026.1 + free license, then from `hls/`
  run `run_synth.bat` → `python parse_hls_report.py` → `python fpga_constants_from_hls.py`
  (details in [`hls/README.md`](hls/README.md)).
- **Calibration LOO:** `python calibration/loo_cv.py` (add rows to
  `calibration/benchmarks.csv` first).

## Status of the grounded FPGA constants (`hls/reports/`)

Synthesized on xc7z020, one representative design point per op. Depthwise,
pointwise, and batchnorm stream near 1 MAC/cycle with trivial DSP; conv is the
dense/expensive case (see `hls/README.md` for the exact numbers and caveats).
