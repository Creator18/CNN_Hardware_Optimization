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

## Results & honesty
Synthesized on xc7z020 (numbers in `reports/`):

| op | MACs/cycle | DSP (of 220) | eff/DSP | note |
|----|-----------|--------------|---------|------|
| conv3x3 | 57.2 | 322 (146%, over-maps) | 0.178 | on-chip cached + zero-padded + IC_UNROLL=4; high throughput but exceeds the 7020 → must be tiled/time-multiplexed in practice (consistent with the paper: dense conv is the FPGA's hard case) |
| depthwise3x3 | 1.00 | 9 | 0.111 | single-lane; streams cleanly |
| pointwise | 1.00 | 16 | 0.062 | single-lane; streams cleanly |
| batchnorm | 0.99 | 1 | 0.994 | near-free, as expected |

The qualitative ordering matches the paper's thesis (BN/depthwise/pointwise
stream efficiently; dense conv is expensive and doesn't fit).

**Two caveats before using these as final cost constants:**
1. **Unmatched parallelism.** conv3x3 is unrolled (IC_UNROLL=4) while the others
   are single-lane, so raw MACs/cycle isn't apples-to-apples. `eff/DSP`
   (throughput per DSP) is the fairer normalized metric, but for a clean
   comparison give all four kernels comparable parallelism.
2. **Sweep the design space.** Vary `IC_UNROLL` (in `kernels/conv3x3.cpp`) and
   report a latency-vs-DSP curve rather than one point — this feeds the
   Monte-Carlo uncertainty analysis (#6) and shows the fit/throughput tradeoff.

Report the design point(s) you actually use in the paper.

Shapes to synthesize live in each `*_top()` and in the `.cfg` files; edit them to
match specific layers from `build_resnet18` / `build_mobilenetv2` in the notebooks.
