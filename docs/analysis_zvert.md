# analysis — new-geometry (Z-vertical) retrain + generative UQ (2026-07-22)

Data: shared numpy re-extracted with Z on axis0 (128) / horizontal on axis1 (64)
— up-samples native 51x126 -> 128x64 instead of down-sampling the vertical.
37 sims, split 125/40/20 functions (unchanged). norm_stats regenerated.

## FAE (Stage 1) — IMPROVED
| metric | old geom (x-on-128) | new geom (Z-on-128) |
|---|---|---|
| full-grid recon R2 [test] | 0.773 | **0.797** |
| full-grid recon R2 [valid]| 0.573 | **0.604** |
| best valid MSE | 0.373 | **0.338** |
Per-slice-pooled recon R2 (eval): **0.835** global. The FAE captures the sharper
vertical structure better -> the re-extraction paid off at the autoencoder level.

## DiT (Stage 2) — REGRESSED (confirmed, 2 retrains)
| readout | old geom | new geom seed42 | new geom seed777 |
|---|---|---|---|
| gen global R2 | 0.808 | 0.683 | 0.652 |
| rf-loss | 0.128 | 0.129 | 0.130 |
Mean 0.668 +/- 0.016 over 2 seeds -> a REAL drop, not seed variance (delta >> +/-0.02
band). Per-slice: Main_external 0.61-0.63, Main_cavity 0.52-0.60 crashed; Wing_external
0.69-0.72, Mid_section 0.81 held. Latent global stats unchanged (sd 1.28 vs 1.27).

### Diagnosis (two effects, both real)
1. **recon<->Lip(D) tradeoff (FunDiff eq.1).** Same rf-loss (0.129) decodes to much
   worse fields than on the old geometry -> the sharper new FAE decoder amplifies
   the DiT's latent errors most on the high-dynamic-range Main slices.
2. **Harder target.** New target carries 128 (was 64) samples in the info-rich
   vertical -> more high-freq structure to reproduce. gen R2 0.668 on the sharper
   target is NOT apples-to-apples with the old 0.808 on the blurrier target.
FAE recon (0.835) shows the ceiling rose; the generator just cannot reach it here.

## Generative UQ (24 samples/case) — informative but overconfident
Both seeds agree: coverage@1sigma ~0.36 (ideal 0.683), @2sigma ~0.61 (ideal 0.954),
sharpness ~33 degC, error<->uncertainty corr **0.61-0.64** (>0 = the std map DOES
flag where the model is wrong), miscalibration area ~0.25. -> The uncertainty is
directionally useful but the bands are too tight (overconfident). This matches
Chavare's finding that raw predictive spreads often need recalibration.

## Next mutations (ranked)
1. **Latent-robustness FAE variant** (highest EV): add a small VAE-style latent
   KL / variance penalty (or spectral norm on the decoder) to lower Lip(D); trade a
   little recon for a lot of gen. One FAE retrain, then reuse the DiT recipe.
2. **Recalibrate the UQ**: temperature-scale the predictive std on the valid split
   (single scalar s minimizing valid NLL) -> should pull coverage toward nominal
   cheaply, no retrain. Report calibrated + raw.
3. Do NOT just enlarge the DiT — rf-loss already converged; the bottleneck is
   decoder sensitivity, not DiT capacity.

## Bookkeeping
Preserved: models_x128_backup/ (old-geom checkpoints), numpy_x128_old/ (old data),
dit_best_seed42.pt (kept as dit_best.pt — better of the two), metrics_seed42.json,
uq_metrics_seed42.json. train_dit.py seed reverted to 42.
