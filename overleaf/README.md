# Overleaf upload package

Self-contained LaTeX source + figures for the paper, including the new
hardware-grounded evaluation (Trial 3).

## Upload
1. Zip this folder (or upload its contents) to a new Overleaf project.
2. Set the main document to `adaptive_cnn_grounded.tex`.
3. Compiler: **pdfLaTeX**. The `llncs` (Springer LNCS) class is included in
   Overleaf's TeX Live distribution — no extra files needed.

## Contents
- `adaptive_cnn_grounded.tex` — the paper, self-contained: includes a
  **Related Work** section and an embedded `thebibliography` (19 references),
  so it compiles in a single pdfLaTeX pass with **no `.bib` / BibTeX step**.
  `\graphicspath{{./}}`, so all images are found in this flat folder.
- 16 `.png` figures referenced by the document (Trials 1–3).

## What changed vs the original template
This is a strict **superset** of the original writeup: all of the original
methodology (roofline cost model, fusion, quantization, partition + multi-split,
functional emulation, predictor, energy model, calibration) and all Trial 1–2
results, tables, and figures (including the break-even / UltraScale+ projection
and the energy model) are retained in full. Added on top:
- New **Section 4 (Trial 3: Hardware-Grounded Evaluation)**: HLS-synthesized FPGA
  costs (Zynq-7020), measured GPU costs (RTX 4070), cross-device calibration,
  a de-circularized leave-one-architecture-out predictor, Monte-Carlo
  uncertainty, a discrete-event pipeline simulation, and an INT8 accuracy axis.
- Abstract, contributions, discussion, and conclusion extended to the
  architecture-scoped, uncertainty-bounded conclusion; the in-model 0.20%
  predictor figure is retained but explicitly contrasted with the grounded
  0–26% leave-one-architecture-out regret.
