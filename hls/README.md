# HLS cost grounding for the FPGA side

These kernels replace the hand-set FPGA efficiency constants of Trial 2 with
numbers obtained from AMD Vitis HLS 2026.1 synthesis for the Zynq-7020 part
(`xc7z020-clg400-1`, 100 MHz). C-synthesis runs on a CPU; no board is needed.

## Files

| Path | Contents |
|------|----------|
| `kernels/conv_tiled.cpp` | Output-stationary DSP-tiled 3x3 convolution (tiles `TM` x `TN`). The CONV cost source. |
| `kernels/conv3x3.cpp` | Naive convolution kept as a baseline; shows the DSP cliff described in the paper. |
| `kernels/{depthwise3x3,pointwise,batchnorm,relu,pool,fc}.cpp` | INT8 streaming kernels for the other operators. |
| `tb/tb_conv.cpp` | C-simulation testbench: checks the tiled conv against a reference convolution. |
| `hls_<name>.cfg` | One `v++` config per design point; output goes to `work/<name>/`. |
| `run_synth.bat` | C-synthesizes all 13 design points that `parse_hls_report.py` reads. |
| `run_impl.bat` | C-simulation plus Vivado place-and-route of the chosen conv (TM=16, TN=8). |
| `parse_hls_report.py` | Synthesis XML to `reports/<op>.json`; picks each operator's best design point that fits the device. |
| `fpga_constants_from_hls.py` | `reports/*.json` to `reports/_fpga_constants.json`, which `trial3.ipynb` reads. |
| `reports/` | Committed results. Build directories (`work/`) are git-ignored. |

## Reproduce

Requirements: Vitis 2026.1 and the free node-locked license (Vivado Basic tier
plus Vitis HLS) pointed to by `XILINXD_LICENSE_FILE`. Edit `VITIS_SETTINGS` at
the top of the `.bat` files if your install path differs. From this folder:

```
run_synth.bat
python parse_hls_report.py
python fpga_constants_from_hls.py
run_impl.bat
```

Toolchain notes for 2026.1: HLS C-synthesis is driven by `v++ -c --mode hls`
(the old `vitis_hls -f` Tcl flow is gone), C-simulation and implementation by
`vitis-run --mode hls --csim / --impl`, and `v++` must be invoked with `call`
inside a `.bat`. Implementation runs in a short directory because Vitis exceeds
the Windows 260-character path limit under a long project path.

## Results (from `reports/`)

Per-operator design point used by the cost model:

| Operator | Design point | MACs/cycle | Efficiency (MACs/cycle / 220) | DSP | Fits xc7z020 |
|----------|--------------|-----------:|------------------------------:|----:|:------------:|
| CONV | conv_tiled 16x8 | 63.03 | 0.2865 | 72 | yes |
| DEPTHWISE_CONV | dw_u1 | 2.93 | 0.0133 | 8 | no (BRAM 292/280) |
| POINTWISE_CONV | pointwise | 7.58 | 0.0344 | 16 | yes |
| BATCHNORM | batchnorm | 2.00 | 0.0091 | 64 | yes |
| RELU | relu | 1.00 | 0.0045 | 0 | yes |
| POOL | pool | 1.76 | 0.0080 | 0 | no (BRAM 514/280) |
| FC | fc_u1 | 1.00 | 0.0045 | 1 | yes |

Effective FPGA memory bandwidth from the memory-bound kernels: 4.996 bytes/cycle.

Tiled conv sweep (`reports/conv_sweep.json`): DSP grows smoothly with `TM x TN`
(no cliff), and on-chip BRAM becomes the binding resource.

| TM x TN | Cycles | DSP | BRAM_18K | Fits |
|---------|-------:|----:|---------:|:----:|
| 8 x 8 | 3,051,634 | 40 | 196 | yes |
| 16 x 8 | 1,834,182 | 72 | 260 | yes |
| 16 x 12 | 1,495,506 | 166 | 388 | no |
| 16 x 16 | 1,370,054 | 136 | 388 | no |

Implementation of the 16 x 8 conv (`reports/conv_impl.json`): C-simulation
passes on all 200,704 outputs; after place-and-route it uses 73 DSP, 11,011 LUT,
7,995 FF and 258 of 280 BRAM_18K, and meets timing at 100 MHz (post-route
critical path 9.866 ns).

## Limits

Each operator is one representative layer shape at one design point. Depthwise
and pool cache the whole input on-chip and exceed BRAM as written; a line-buffer
design would fit. Only the conv has been place-and-routed; the other operators
are C-synthesis estimates.
