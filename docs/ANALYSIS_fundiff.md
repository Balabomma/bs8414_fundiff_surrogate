# FunDiff surrogate — design & lineage

**Parent:** new branch — not a mutation of an existing `bs8414_*` variant. Shares
the slice-field target + 42/9/9 split with `bs8414_slice_surrogate` /
`bs8414_physicsnemo_surrogate` / `samba_MLP`, and reuses the PhysicsNeMo
project's extracted `numpy/` data read-only.

**Source paper:** FunDiff — Wang, Dou, Shan, Liu & Lu, *Nature Communications*
17:5749 (2026), doi:10.1038/s41467-026-72292-0. PDF: `D:\VS_projects\s41467-026-72292-0.pdf`.

## The one idea
Generate the temperature field as a **continuous spatiotemporal function**
`T(t,y,z)` via FunDiff's Function-Autoencoder + latent rectified-flow DiT, rather
than regressing a fixed grid. Predicted effect vs the FNO/Transolver and Samba
slice surrogates: (a) arbitrary temporal/spatial super-resolution from one model,
(b) physics priors enforced in continuous space in the FAE decoder (hard ambient
floor + soft growth-monotonicity), decoupled from the generative stage.

## Decisions locked with the user (2026-07-20)
- **Staged full FunDiff.** Stage 1 = FAE (usable standalone as a deterministic
  comparison entry, reported as `FAE(recon)` R²); Stage 2 = latent DiT
  (rectified flow, AdaLN-Zero) conditioned on the 16-d params + slice id,
  read out deterministically as the conditional mean (`DiT(gen)`).
- **Spatiotemporal 3D functions.** One function = one (sim, slice); ~185 total
  (train 125 / valid 40 / test 20 present).
- **New directory + Smokeview output** (reuses sibling `pysmokview`).

## Config diff vs the paper's implementation defaults
- FAE: 8-layer enc / 4-layer dec, D=256, MLP=512, 8 heads (paper's defaults),
  N=128 Perceiver latents; 3D patchify (8,16,16) with T padded 181→184.
- DiT: 8 blocks, D=256, MLP=512, 8 heads (paper's defaults).
- Params: FAE ≈ 7.5M, DiT ≈ 7.8M (≈ 15.3M total).
- LR schedule: warmup 2000 steps → exp decay 0.9 / 2000 steps, AdamW wd 1e-5
  (paper's protocol). FAE batch 8 (paper 16; 8 fits the 181-frame fields on the 4090).

## Known risks / hypotheses to test after the first runs
1. **Data-starved DiT** — ~125 training functions is small for a generative
   model. Watch the `FAE(recon)` → `DiT(gen)` R² gap: a large gap = the DiT can't
   reach the FAE ceiling → try classifier-free-guidance-style dropout of the
   condition, or more `N_COND_SAMPLES`, or a deterministic (mean-latent) target.
2. **FAE ceiling** — if `FAE(recon)` R² itself is low, the Perceiver bottleneck
   (128 tokens) is too tight → raise `N_LATENT_TOKENS` before touching the DiT.
3. **Growth prior** — soft; if growth-phase monotonicity fails on test, move it
   to a harder parameterization (cumulative-softplus in time) in the decoder.

## Smoke tests (passed, GPU, torch 2.11+cu128, RTX 4090)
- FAE fwd/bwd on real (B,181,128,64) fields; ambient floor holds.
- Off-grid super-res decode (362×32×16) works.
- DiT rectified-flow loss + backward; sampler sign check PASS (|x(1)−x1|≈2e-6).
- End-to-end real-data `fae_step` (mse 1.41, ssim 1.07, growth 0.018, energy 0.53
  at init) and DiT step on real latents.

## Next actions
- `python train_fae.py` → record `analysis_fae_run1.md` (recon R² per slice vs
  the FNO baseline's ±0.02 band, physics sanity).
- Then `python train_dit.py` → `analysis_dit_run1.md` (gen vs recon gap).
- Preserve `models/` to a named dir before any retrain (VARIANCE_RECORD rule).
