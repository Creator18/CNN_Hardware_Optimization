# Adaptive CNN Acceleration on GPU and FPGA

Code, results and paper draft for complexity-aware partitioning of CNN inference
across an edge GPU and an FPGA.

Angshuman Basu, Northeastern University. Status: paper draft, not yet submitted.

## The idea

Edge boards often have both a GPU and an FPGA, yet a CNN usually runs on only one
of them. CNN layers behave very differently: dense 3x3 convolutions are
compute-heavy and suit a GPU, while depthwise convolutions, batch normalization,
ReLU and pooling are memory-bound and suit an FPGA streaming datapath.

This project splits a network into a GPU stage and an FPGA stage, fuses
consecutive layers (conv + BN + ReLU), quantizes the FPGA side to INT8, and
pipelines the two stages so both devices work on consecutive frames. A cost model
chooses the split point. The question is whether this beats the GPU alone, and the
project answers it at three increasing levels of rigour:

1. **Trial 1, analytical:** roofline cost model with hand-set per-operator efficiencies.
2. **Trial 2, emulation:** device emulators running real tensor operations, calibrated against published benchmarks.
3. **Trial 3, hardware-grounded:** FPGA costs from AMD Vitis HLS synthesis for the Zynq-7020 (one kernel taken through place-and-route), GPU costs measured on an RTX 4070 laptop GPU, plus cross-device calibration, a discrete-event pipeline simulation, Monte-Carlo uncertainty and an INT8 fidelity check.

## Results

Throughput gain of the GPU + FPGA pipeline over GPU-only (Jetson Nano class GPU,
Zynq-7020 FPGA, PCIe link):

| Level | ResNet-18 | MobileNetV2 |
|-------|----------:|------------:|
| Trial 1, analytical | +0.8% | +11.8% |
| Trial 2, emulated | +75.0% | +10.1% |
| Trial 3, grounded, relative (unit space) | +155.2% | +102.2% |
| Trial 3, after discrete-event pipeline simulation | +132.0% | +81.0% |
| Trial 3, calibrated to absolute time on the Zynq-7020 | **-79.0%** | **+52.1%** |

How to read this table:

- The relative (unit-space) rows show that the per-operator cost structure favours
  splitting. They assume a GPU cost unit and an FPGA cost unit take the same time,
  which is not true.
- Calibrated to milliseconds, only MobileNetV2 is faster on today's Zynq-7020.
  ResNet-18 is slower because the device is too slow in absolute terms for its
  heavier convolution load. The emulation-level break-even analysis puts the
  required FPGA speed at 1.5x to 3x a Zynq-7020.
- Monte-Carlo over the synthesized constants (3000 samples) keeps the relative
  gains positive: ResNet-18 +155.2% (90% CI +107.5% to +225.0%), MobileNetV2
  +101.2% (90% CI +64.4% to +151.8%).

Other findings:

- **The FPGA convolution datapath decided the conv-heavy result.** A naive HLS
  convolution hit a DSP cliff. An output-stationary DSP-tiled kernel
  (`hls/kernels/conv_tiled.cpp`) reaches 63.03 MACs/cycle on 72 DSPs. It passes
  C-simulation on all 200,704 outputs and, after place-and-route on the xc7z020,
  uses 73 DSP and 258 of 280 BRAM blocks and meets timing at 100 MHz.
- **Inference is memory-bound on both devices.** Depthwise, BN, ReLU and pooling
  do not get faster with more DSPs.
- **Measured GPU efficiency is far more skewed than the hand-set values.**
  Normalised to the best operator: conv 1.0, pointwise 0.1167, depthwise 0.0596,
  pool 0.0515, BN 0.0069, ReLU 0.0068, FC 0.0067.
- **Latency predictor.** Trial 1 reported 0.20% scheduling regret, but that
  predictor was trained and tested on the same analytical formula. Under
  leave-one-architecture-out evaluation on synthesized targets, regret is 12.34%
  to 36.99% across ResNet-18, MobileNetV2, VGG-11 and EfficientNet-Lite.
- **INT8 fidelity.** With all weights in INT8, top-1 agreement with FP32 is 100%
  for ResNet-18 and 94% for MobileNetV2.

Caveats:

- Nothing ran on a physical FPGA board or Jetson. FPGA costs are synthesis
  results; only the convolution was place-and-routed. The depthwise and pooling
  kernels exceed on-chip BRAM as written.
- GPU efficiencies come from a laptop RTX 4070 at batch size 1, not the Jetson
  target, so only the relative pattern carries over.
- Calibration uses two published anchors per device
  (`calibration/benchmarks.csv`); one GPU anchor is an extrapolated estimate and
  is excluded from calibration.
- The INT8 check fake-quantizes weights only, on random inputs. It measures
  agreement with FP32, not ImageNet accuracy.
- The pipeline simulation does not model shared-memory contention.

## Repository layout

```
trial1.ipynb               Trial 1: analytical model, 6 architectures
trial2.ipynb               Trial 2: functional emulation and calibration
trial3.ipynb               Trial 3: hardware-grounded evaluation
ResultImages_testingPhase/   figures from Trial 1
ResultImages_emulationPhase/ figures from Trial 2
ResultImages_groundedPhase/  figures from Trial 3
hls/                       HLS kernels, configs, synthesis and implementation scripts, results
gpu/                       GPU per-operator profiler and results
calibration/               published benchmark anchors and a leave-one-out harness
overleaf/                  paper source (Springer LNCS) and the figures it uses
Adaptive_CNN_Acceleration.pdf  earlier draft covering Trials 1 and 2 only
```

## Reproduce

```
pip install -r requirements.txt
```

- **Notebooks:** run `trial1.ipynb`, `trial2.ipynb` and `trial3.ipynb` top to
  bottom in Jupyter. `trial3.ipynb` reads the committed `hls/reports/` and
  `gpu/reports/`, so it runs without Vitis or a GPU; its accuracy cell downloads
  pretrained torchvision weights. The local-GPU timing cell of `trial2.ipynb`
  needs a CUDA GPU.
- **FPGA synthesis:** see [`hls/README.md`](hls/README.md) (Vitis 2026.1 and a
  free license; no board needed).
- **GPU profile:** `python gpu/profile_gpu.py` (CUDA GPU) rewrites
  `gpu/reports/_gpu_constants.json`.
- **Paper:** upload the `overleaf/` folder to Overleaf and compile
  `adaptive_cnn_grounded.tex` with pdfLaTeX.
