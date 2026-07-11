# Calibration benchmark dataset

`benchmarks.csv` collects **published** (and a few locally-measured) CNN
inference latencies on our two target device classes, so calibration can be
tested by **leave-one-out cross-validation** instead of fitting one factor per
point (the current weakness: 2 anchors per device ⇒ near-zero error by
construction).

## Columns
| column | meaning |
|--------|---------|
| `source` | short citation key (author/year or repo) |
| `url` | link to the paper/repo (for the bibliography) |
| `model` | CNN architecture (must match a `build_*` in the notebooks) |
| `device` | `jetson_nano` \| `zynq_7020` \| `rtx4070` \| ... |
| `device_class` | `GPU` \| `FPGA` |
| `precision` | `FP32` \| `FP16` \| `INT8` \| `INT16` \| ... |
| `framework` | TensorRT / TF / custom VHDL / HLS / PyTorch ... |
| `input_size` | e.g. `224x224`, `256x256` |
| `latency_ms` | single-image (batch=1) latency in ms |
| `throughput_fps` | if reported (else blank) |
| `gmac_per_s` | achieved GMAC/s if reported (FPGA papers often do) |
| `is_estimate` | `1` if extrapolated/guessed rather than directly measured |
| `notes` | anything caveat-worthy |

## Rules
- One row per (source, model, device, precision) measurement.
- Prefer **batch=1** numbers (edge streaming scenario).
- Mark anything not directly measured with `is_estimate=1` (e.g. the current
  MobileNetV2 Jetson figure is a TRT-speedup extrapolation — flagged so LOO can
  optionally exclude estimates).
- Add rows liberally: more Zynq-7020 / Jetson Nano CNN papers = a stronger LOO
  test. Target ≥ 6–8 real points per device class.
