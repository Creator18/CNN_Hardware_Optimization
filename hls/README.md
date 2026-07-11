# HLS FPGA cost grounding

Goal: replace the **hand-set FPGA efficiency constants** in `trial2.ipynb`
(`streaming_efficiency = {CONV: 0.45, DEPTHWISE_CONV: 0.85, ...}`) with numbers
**derived from Vitis HLS synthesis** for the actual Zynq-7020 (`xc7z020`).
No board required — C-synthesis runs on CPU.

## Files
| File | What it is |
|------|-----------|
| `kernels/common.h` | shared INT8 types + representative layer shapes |
| `kernels/conv3x3.cpp` | standard 3x3 conv (compute-heavy) |
| `kernels/depthwise3x3.cpp` | depthwise 3x3 (FPGA's claimed strength) |
| `kernels/pointwise.cpp` | 1x1 conv |
| `kernels/batchnorm.cpp` | elementwise BN (near-free streaming) |
| `hls_<kernel>.cfg` | v++ config per kernel (part, top, clock) |
| `run_synth.bat` | runs `v++ -c --mode hls` for all four kernels |
| `parse_hls_report.py` | report XML → `reports/<kernel>.json` |
| `fpga_constants_from_hls.py` | `reports/*.json` → FPGA efficiency constants |
| `reports/` | **committed** extracted JSON (build dirs are gitignored) |

## Toolchain note (2026.1)
The classic `vitis_hls -f script.tcl` wrapper was removed; under `vitis-run` the
Tcl `add_files` path handling is broken. The working flow is the unified
compiler: **`v++ -c --mode hls --config <kernel>.cfg --work_dir work/<kernel>`**.
`run_synth.bat` wraps this. You also need the free node-locked license
(Vivado **Basic** tier + Vitis HLS, both No-Charge) at `XILINXD_LICENSE_FILE`.

## Workflow
1. From this `hls/` directory in a cmd prompt:
   ```
   run_synth.bat
   ```
   (~1-3 min per kernel; writes `work/<kernel>/hls/syn/report/*_csynth.xml`.)
2. Extract the numbers:  `python parse_hls_report.py`  → `reports/<kernel>.json`
   (latency cycles + µs, initiation interval, DSP/BRAM/FF/LUT + utilization).
3. Derive the constants: `python fpga_constants_from_hls.py` →
   `reports/_fpga_constants.json`.
4. Commit **only** `reports/*.json` (the `work/` build dirs are gitignored),
   then import `_fpga_constants.json` into `trial2.ipynb` to replace the
   hand-set `streaming_efficiency` values.

## First-run results & honesty
The current kernels are **naive representative datapaths**, not optimized
accelerators, and the numbers show it:

| op | MACs/cycle | DSP (xc7z020 has 220) | note |
|----|-----------|------------------------|------|
| conv3x3 | 0.08 | 575 (261%, over-maps) | memory-port bound as written — needs line buffers / array partitioning / tiling for a fair number |
| depthwise3x3 | 1.00 | 9 | streams cleanly |
| pointwise | 1.00 | 16 | streams cleanly |
| batchnorm | 0.99 | 1 | near-free, as expected |

The qualitative ordering already matches the paper's thesis (BN/depthwise stream
efficiently; dense conv is the hard case). But before using these as cost
constants, **tune the conv kernel** (add local buffering so it isn't bound by
`m_axi` port contention) and ideally **sweep pragmas** to report a
latency-vs-DSP range — which pairs naturally with the Monte-Carlo uncertainty
analysis (#6) in the notebook. Report the design point(s) you actually use.

Shapes to synthesize live in each `*_top()` and in the `.cfg` files; edit them to
match specific layers from `build_resnet18` / `build_mobilenetv2` in the notebooks.
