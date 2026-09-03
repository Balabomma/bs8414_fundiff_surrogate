# analysis — FAE run1 + DiT run1 (2026-07-22)

Split: fixed 42/9/9 hash (present: train 125 / valid 40 / test 20 functions).
Logs: train_fae_run1.log, train_dit_run1.log, evaluate_run1.log.
Checkpoints: models/fae_best.pt, models/dit_best.pt.

## Config diff vs parent (paper defaults)
- FAE_WARMUP_STEPS 2000 -> 300 (paper's 2000 STEPS = ~125 epochs here; too long on
  16 steps/epoch). One change. All else per config.py.

## Results — test set, physical degC
| Slice | FAE(recon) R2 | DiT(gen) R2 | RMSE(gen) | SSIM(gen) |
|---|---|---|---|---|
| Main_external | 0.834 | 0.831 | 96.1 | 0.537 |
| Wing_cavity   | 0.666 | 0.660 | 65.9 | 0.642 |
| Wing_external | 0.696 | 0.663 | 67.8 | 0.637 |
| Main_cavity   | 0.813 | 0.804 | 102.1| 0.534 |
| Mid_section   | 0.845 | 0.829 | 78.6 | 0.654 |
| **GLOBAL**    | **0.817** | **0.808** | **83.4** | — |

FAE full-grid recon R2 at train time: test 0.773 / valid 0.573.
DiT best valid rf-loss 0.128 (3000 ep, no early stop).

## Findings
1. **DiT reaches the FAE ceiling.** recon->gen global gap = 0.009 R2 (inside the
   +/-0.02 band). Hypothesis (risk #1 in ANALYSIS_fundiff.md — data-starved DiT)
   did NOT bite: strong param+slice conditioning collapsed the generator to a
   near-deterministic map on 125 functions. So the bottleneck is the FAE, not the DiT.
2. **FAE ceiling is the limiter.** Peak turbulent cores under-predicted by
   400-700 degC (see *_cmp.png); fields are smooth/mean-reverting. Consistent
   with the 128-token Perceiver bottleneck + softplus floor.
3. **Physics sanity PASS**: ambient floor holds (no sub-ambient), plume envelope
   and location correct, cold background preserved. Wing slices weakest (0.66-0.70)
   — same hard slices the other surrogates up-weight.

## Geometry / rendering correction (2026-07-22)
Found via user feedback that the Smokeview axes were pixel indices, not metres.
Verified from fdsreader `slc.extent` + burner location that the shared extraction
resized native (X x Z) slices to (128, 64) mapping **axis0(128)=in-plane
HORIZONTAL (4 m)** and **axis1(64)=VERTICAL Z (10 m)** — the opposite of the
`FIELD_HEIGHT=vertical` label. All 5 slices are 4 m x 10 m (domain 4x4x10).
Fixed `smokeview_output.py` to transpose to Z-up, put true metres on both axes,
equal aspect, and annotate the real cutting plane; added `SLICE_GEOMETRY` +
`STORED_AXIS0_IS_HORIZONTAL` to config.py.
- **Scope:** this is a SHARED-pipeline issue — every slice surrogate
  (bs8414_slice_surrogate, physicsnemo, samba) reads the same numpy and renders
  the same transposed/pixel axes. Their METRICS are unaffected (orientation- and
  unit-invariant), only their figures. Flagged to the user; siblings not edited.
- **Also note:** the 128x64 target over-samples the 4 m horizontal (0.031 m/px)
  and under-samples the 10 m vertical (0.156 m/px) — vertical (the buoyancy
  direction with the most structure) is the coarser one. A future extraction
  should use (Z=128, horiz=64) or an aspect-matched grid.

## Next mutations (exploit the FAE, since the DiT already saturates it)
- **Raise N_LATENT_TOKENS 128 -> 256/384** (loosen the bottleneck) — highest-EV
  single change; re-check recon R2 before retraining the DiT.
- Add a **high-frequency reconstruction term** (gradient/Laplacian match or
  perceptual) to fight the peak smoothing; or CNO-style bandlimited decoder head.
- Only after the FAE ceiling rises: revisit the DiT (it currently has slack).
- Append a comparison row vs FNO/slice/samba on this split to comparison_logs/
  before any "better/worse" claim (variance rule — 0.009 is not a win by itself).
