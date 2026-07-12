# GPU cost grounding (C7)

Companion to `hls/` (which grounds the FPGA side). Measures the **per-op GPU
efficiency** — how well each layer type uses the GPU at batch=1 — on the local
CUDA GPU, replacing the hand-set `parallel_efficiency` in trial2's `GPUEmulator`.

## Run
```
python gpu/profile_gpu.py
```
Requires PyTorch with CUDA. Writes `gpu/reports/_gpu_constants.json`
(per-op `efficiency`, `latency_us`, `achieved_gmac_s`).

## Method
Each op type is run as a real `torch` module at **batch=1** on the representative
shapes used by the HLS kernels (so the two studies line up), timed with CUDA
events (warmup + median over 200 runs). Per-op `efficiency = achieved GMAC/s /
peak GMAC/s`, where peak = the best op (dense conv) — i.e. how close each op gets
to the device's realized peak, the same 0..1 knob the emulator uses.

## Result (RTX 4070 Laptop, FP32, batch=1)
| op | efficiency | vs trial2 hand-set |
|----|-----------|--------------------|
| conv | 1.00 | 0.65 |
| pointwise | 0.12 | 0.55 |
| depthwise | 0.06 | 0.20 |
| pool | 0.05 | 0.12 |
| batchnorm | 0.007 | 0.15 |
| relu | 0.007 | 0.10 |
| fc | 0.007 | 0.50 |

The GPU is peak-efficient on dense conv and *far* weaker elsewhere than hand-set
assumed — at batch=1 the small/memory-bound ops are kernel-launch-bound
(~10 GMAC/s floor). Realistic for edge single-image streaming, but note the
absolute efficiency of trivial ops is launch-bound.

## Caveats
- Measured on the local GPU (not the Jetson target): the **relative** per-op
  pattern transfers; absolute times do not.
- **Do not** fold these into a both-devices-grounded gain without cross-device
  time calibration — GPU efficiency (normalized to GPU peak) and FPGA efficiency
  (normalized to 220 MAC/cycle) are on different scales, so the unit-space ratio
  is meaningless until calibrated. trial3's GPU cell demonstrates this
  (both-grounded gains explode to +1400%-class); the headline keeps the GPU
  hand-set to isolate the FPGA grounding.
