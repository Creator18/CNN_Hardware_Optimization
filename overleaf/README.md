# Overleaf upload package

Self-contained LaTeX source + figures for the paper, including the new
hardware-grounded evaluation (Trial 3).

## Upload
1. Zip this folder (or upload its contents) to a new Overleaf project.
2. Set the main document to `adaptive_cnn_grounded.tex`.
3. Compiler: **pdfLaTeX**. The `llncs` (Springer LNCS) class is included in
   Overleaf's TeX Live distribution — no extra files needed.

## Contents
- `adaptive_cnn_grounded.tex` — the paper. `\graphicspath{{./}}`, so all images
  are found in this flat folder.
- 15 `.png` figures referenced by the document (Trials 1–3).

## What changed vs the previous writeup
- New **Section 4 (Trial 3: Hardware-Grounded Evaluation)**: HLS-synthesized FPGA
  costs (Zynq-7020), measured GPU costs (RTX 4070), cross-device calibration,
  a de-circularized leave-one-architecture-out predictor, Monte-Carlo
  uncertainty, a discrete-event pipeline simulation, and an INT8 accuracy axis.
- Abstract, contributions, discussion, and conclusion updated to the
  architecture-scoped, uncertainty-bounded conclusion and to correct the
  in-model 0.20% predictor figure (grounded regret is 0–26%).
