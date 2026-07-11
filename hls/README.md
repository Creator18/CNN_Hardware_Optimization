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
| `run_synth.tcl` | synthesizes all four for `xc7z020`, 100 MHz |
| `parse_hls_report.py` | report XML → `reports/<kernel>.json` |
| `fpga_constants_from_hls.py` | `reports/*.json` → FPGA efficiency constants |
| `reports/` | **committed** extracted JSON (build dirs are gitignored) |

## Workflow (once Vitis HLS is installed)
1. Open the **Vitis HLS Command Prompt** (Windows Start menu, installed with the
   toolchain) so `vitis_hls` is on PATH. From this `hls/` directory:
   ```
   vitis_hls -f run_synth.tcl
   ```
   This creates `projects/<kernel>/` and runs C-synthesis for each kernel.
   (First run may take several minutes per kernel.)
2. Extract the numbers:
   ```
   python parse_hls_report.py
   ```
   Writes `reports/conv3x3.json`, etc. — latency (cycles + µs), initiation
   interval, DSP/BRAM/FF/LUT and utilization %.
3. Derive the cost-model constants:
   ```
   python fpga_constants_from_hls.py
   ```
   Writes `reports/_fpga_constants.json`.
4. Commit **only** `reports/*.json` (the `projects/` build dirs are gitignored),
   then import `_fpga_constants.json` into `trial2.ipynb` to replace the
   hand-set `streaming_efficiency` values.

## Tuning / honesty
The kernels are simple representative datapaths, not optimized accelerators.
Different `PIPELINE` / `UNROLL` / `ARRAY_PARTITION` choices land on different
latency-vs-DSP points. Either (a) report the single design point you use, or
(b) sweep pragmas and report the range — the latter pairs well with the
Monte-Carlo uncertainty analysis (#6) back in the notebook.

Shapes to synthesize live in each `*_top()` and in `run_synth.tcl`; edit them to
match the specific layers you want to ground (see `build_resnet18` /
`build_mobilenetv2` in the notebooks).
