# bs8414_fundiff_surrogate — FunDiff slice-field surrogate

A BS 8414-1 facade-fire **slice-field** surrogate built on **FunDiff** (Wang, Dou,
Shan, Liu & Lu, *"FunDiff: diffusion models over function spaces for
physics-informed generative modeling"*, Nature Communications **17**:5749, 2026,
[doi:10.1038/s41467-026-72292-0](https://doi.org/10.1038/s41467-026-72292-0)).

One "function" is the **whole spatiotemporal field `T(t, y, z)` for one
(simulation, slice)** — so the decoder gives arbitrary temporal *and* spatial
super-resolution, and the diffusion stage works in a compact latent space rather
than on 1.48 M grid points.

Everything below belongs to either the **60-sim corpus** (the original
cladding × HRR × mesh study) or the **Part1 geometry corpus** (185 geometry
variants). They share neither target nor split — never mix their numbers. The
`PART1_MODEL_DIR` environment variable is what keeps a Part1 run from
overwriting 60-sim weights.

| | 60-sim corpus | Part1 geometry corpus |
|---|---|---|
| Data layer | `data_processing/dataset.py` | `dataset_part1.py` → `slice_loader_part1.py` |
| Config | `config.py` | `config.py` (architecture) + `config_part1.py` (corpus) |
| Design axes | cladding(4) × HRR(5) × mesh(3) | cladding(12) × insulation(5) × geometry(8) |
| Conditioning | `ParamConditioner` (16-d params + slice id) | `Part1Conditioner` → 49-d, via `Part1ParamConditioner` |
| Cases / split | 60 sims, 42/9/9 | 184 usable, hash split **141 / 20 / 23** |
| Slice planes | 5 | 5 for cavity cases, **3 for `noair`** (masked, never imputed) |
| Checkpoints | `models/`, `models_x128_backup/` | `models_part1_fundiff_*` |

---

## The models

### Stage 1 — Function Autoencoder (`model/fae.py`)

3D ViT patchify → Perceiver latent bottleneck → self-attention encoder;
**CViT** continuous cross-attention decoder queried at arbitrary `(t, y, z)`.
**7,526,401 params.** Usable standalone as a deterministic surrogate, and its
reconstruction R² is the upper bound Stage 2 is trying to reach.

| FunDiff component | Here |
|---|---|
| Resolution-invariant encoder | trainable positional embeddings, interpolated under temporal sub-sampling augmentation (`FAE_ENC_SUBSAMPLE_T = [1, 2, 4]`) |
| Physics priors in the decoder | hard **ambient floor** (`T = 18 °C + softplus(·)`), soft **growth-phase monotonicity** (`LAMBDA_GROWTH`) and **energy/HRR** (`LAMBDA_ENERGY`) priors on decoded points |
| Reconstruction loss | MSE on random query points + SSIM on decoded full frames (`LAMBDA_SSIM = 0.15`) |
| Latent | `N_LATENT_TOKENS = 128` × `EMBED_DIM = 256` |

### Stage 2 — latent rectified-flow DiT (`model/dit.py`, `model_part1.py`)

8-block Diffusion Transformer with **AdaLN-Zero** modulation, trained by a
**rectified-flow** objective (paper eq. 9) over the *frozen* FAE latents.
Latents are precomputed and cached on the GPU, so Stage 2 is cheap.

| Variant | Class | Conditioning | Params |
|---|---|---|---|
| 60-sim | `model.dit.DiT` | `ParamConditioner`: 16-d param vector + slice embedding | — |
| **Part1** | `model_part1.Part1DiT` (aliased `Part1Surrogate`) | `Part1ParamConditioner`: `Part1Conditioner` (12+8+16 embeddings + 13 material → 49-d) + slice embedding | **7,815,640** |

`MODEL_NAME = "FunDiff DiT (Part1)"`, `LAMBDA_REG = 0.0` — rectified flow carries
no extra weight penalty, so `regularization()` returns a true zero rather than a
silent no-op.

**Inference is a deterministic conditional-mean readout.** `P(field | params)` is
near-deterministic for a fixed FDS deck, so evaluation and the app both sample
`N_COND_SAMPLES = 8` latents from fixed-seed noise (`INFERENCE_SEED = 1234`),
average them, and decode the full 181 × 128 × 64 grid with
`N_SAMPLING_STEPS = 50` midpoint-Euler ODE steps. Changing the seed, the sample
count or the step count changes the readout — all three are recorded.

---

## Layout

```
config.py                   architecture + 60-sim corpus; MODEL_DIR honours $PART1_MODEL_DIR
config_part1.py             Part1 corpus contract (shared, byte-identical across projects)
data_processing/dataset.py  60-sim data layer
dataset_part1.py            Part1 adapter — same build_datasets()/collate() contract
slice_loader_part1.py       reads the shared part1_slices/*.npz          (shared file)
data_loader_part1.py        Part1 CHID/material/split logic              (shared file)
part1_conditioning.py       Part1Conditioner: ids + material -> 49-d     (shared file)
physics_part1.py            physics gates, optional closure/geom penalty (shared file)
model/fae.py                Stage-1 Function Autoencoder
model/dit.py                Stage-2 DiT, rectified-flow loss, latent sampler
model/physics.py            growth-monotonicity + energy/HRR priors
model_part1.py              THE VARIABLE UNDER TEST — Part1DiT + Part1ParamConditioner
train_fae.py                Stage 1 trainer
train_dit.py                Stage 2 trainer
evaluate.py                 frozen eval contract: FAE(recon) and DiT(gen), per slice
evaluate_peak.py            adds the peak-error statistic without touching evaluate.py
evaluate_uq.py              generative UQ: per-pixel predictive mean + std
recalibrate.py              post-hoc Gaussian-NLL std scaling, fitted on valid only
uq_metrics.py               Chavare (arXiv:2607.18294) calibration metrics
smokeview_output.py         Smokeview-style GIF/PNG renders via the sibling pysmokview
explain_part1.py            SHAP attribution                            (shared file)
causal_part1.py             interventional/causal explainability        (shared file)
verify_parity_part1.py      SHA-256 + array-hash proof of shared-layer identity
app_common_part1.py         shared Streamlit input layer (build-up -> 16-d vector)
app_fundiff_part1.py        the Streamlit slice app
run_app.ps1                 app launcher (venv, material table, GPU check, streamlit)
app_assets/                 part1_materials.json + selected_model.json
```

**Inert copies.** `train_part1.py`, `evaluate_part1.py`, `train_slices_part1.py`,
`evaluate_slices_part1.py` and `slice_losses_part1.py` are the *sensor* and
*deterministic-slice* contracts. They are not in this project's parity manifest
(`verify_parity_part1.py` lists them for `SENSOR_PROJECTS` / `SLICE_PROJECTS`
only) and they do not run here — FunDiff trains by rectified flow in two stages,
so it keeps its own trainers. The empty `models_part1_slice_r1` in the sibling
FunDiff-KAN project is the residue of one such attempt.

---

## Training

Always from this directory, in **this project's own venv**, on the NVIDIA GPU.

```powershell
cd D:\VS_projects\bs8414_fundiff_surrogate
.\venv\Scripts\activate
python verify_parity_part1.py          # shared layer identical across projects
```

### Part1 (current work)

`train_fae.py` and `train_dit.py` already import `dataset_part1`, so they train
on the Part1 corpus. The **output directory is set by an environment variable**,
because both scripts write fixed filenames (`fae_best.pt`, `dit_best.pt`):

```powershell
$env:PART1_MODEL_DIR = "models_part1_fundiff_r3"
$env:FUNDIFF_SEED    = "43"                     # default 42
python -u train_fae.py > train_fae_r3.log 2> train_fae_r3.err.log   # Stage 1, ~2.6 h
python -u train_dit.py > train_dit_r3.log 2> train_dit_r3.err.log   # Stage 2, ~25 min
python evaluate.py
Remove-Item Env:PART1_MODEL_DIR, Env:FUNDIFF_SEED
```

Bash equivalent — how the replicate scripts at the root actually drive it:

```bash
d=models_part1_fundiff_r3
FUNDIFF_SEED=43 PART1_MODEL_DIR="$d" ./venv/Scripts/python.exe train_fae.py
FUNDIFF_SEED=43 PART1_MODEL_DIR="$d" ./venv/Scripts/python.exe train_dit.py
PART1_MODEL_DIR="$d" ./venv/Scripts/python.exe evaluate.py
```

`PART1_MODEL_DIR` is not optional discipline — leave it unset and Stage 1 writes
`models/fae_best.pt` straight over the 60-sim checkpoints. Set it before *both*
stages and before evaluation; Stage 2 loads the FAE from the same directory.

Other environment knobs: `PART1_SPLIT` (`hash` default, or `system` for the
unseen-build-up protocol — the two are **not** comparable), `PART1_SIMS_DIR`,
and `PART1_SLICE_DIR` (defaults to the shared
`bs8414_slice_surrogate/data/part1_slices`, 185 cases, ~1 GB — never re-extract
per project).

Reusing a Stage-1 FAE for a Stage-2-only experiment is legitimate and cheap:
copy `fae_best.pt` into the new directory first, then run `train_dit.py` alone.

### 60-sim (legacy)

Swap the data-layer import in `train_fae.py` / `train_dit.py` back to
`from data_processing.dataset import build_datasets, collate` and leave
`PART1_MODEL_DIR` unset. Reads the already-extracted PhysicsNeMo `numpy/` set
read-only through `SHARED_DATA_DIR`.

### Runs on disk

| Directory | Corpus | Seed | Test DiT(gen) R² | RMSE | FAE recon R² |
|---|---|---|---|---|---|
| **`models_part1_fundiff_r1`** ★ | Part1, hash | 42 | **0.8658** | 73.0 °C | 0.8763 |
| `models_part1_fundiff_r2` | Part1, hash | 42 | 0.8644 | 73.4 °C | 0.8778 |
| `models_part1_fundiff_s43` | Part1, hash | 43 | not evaluated | — | — |
| `models/`, `models_x128_backup/` | 60-sim | — | see `analysis_run1.md`, `analysis_zvert.md` | | |

★ = the deployed default, recorded in `app_assets/selected_model.json`. The
r1/r2 margin is **0.0014**, far inside the ±0.02 inconclusive band: best
available, not significantly best.

`models_x128_backup/` predates the **2026-07 geometry fix** (extraction now
transposes so axis0 = 128 = vertical Z); its numbers are not comparable to
anything current. Do not delete it — it is the before-state of that fix.

---

## Evaluation

```powershell
$env:PART1_MODEL_DIR = "models_part1_fundiff_r1"
python evaluate.py                          # -> metrics.json + outputs/fields/*.npz
python evaluate_peak.py --model-dir models_part1_fundiff_r1
python evaluate_uq.py                       # per-pixel mean/std, UQ_N_SAMPLES=24
python recalibrate.py                       # std scale fitted on VALID, applied to test
python smokeview_output.py [--mp4] [--uq]   # -> outputs/smokeview/*.gif, *_cmp.png
```

`evaluate.py` reports **two readouts** so autoencoder quality and generative
quality stay separable:

- **`FAE(recon)`** — encode ground truth → decode. The Stage-1 ceiling.
- **`DiT(gen)`** — sample latent → decode. Full FunDiff.

Metrics are R², RMSE, MAE, MBE, MAPE, p95 and SSIM in physical °C, global and per
slice plane. `metrics.json` is written into the model directory, which is what
the root-level `select_best_model.py` ranks on.

---

## Streamlit app

```powershell
cd D:\VS_projects\bs8414_fundiff_surrogate
.\run_app.ps1                 # http://localhost:8501
.\run_app.ps1 -Port 8502      # alongside the FunDiff-KAN app for a side-by-side
```

`run_app.ps1` activates this venv, picks `app_fundiff_part1.py`, exports the
material table if it is missing, prints GPU status, then starts Streamlit. Manual
equivalent: `.\venv\Scripts\activate ; streamlit run app_fundiff_part1.py`.

Pick a **cladding × insulation × geometry** build-up and it generates the slice
fields: ~4 s per plane on the 4090 at the frozen 50 steps / 8 samples. Tabs are a
time-slider field snapshot on a common colour scale, peak/mean time series, the
vertical temperature envelope, and `.npz` export carrying the fields plus the
settings that produced them. ODE steps, sample count and seed are exposed
because changing them changes the readout.

**Prediction only** — the app never reads `D:\Bs8414_05052026` or the extracted
`part1_slices`. Its runtime inputs are `fae_best.pt` + `dit_best.pt` (the
normalisation stats ride inside them) and `app_assets/part1_materials.json`,
written once by the root-level `export_app_assets.py`, which refuses to write
unless a cladding/insulation id provably fixes its material block.

**Part1 enforced from the checkpoint, not the filename.** A run is offered only
if its `dit_best.pt` carries `cond.conditioner.*` keys — the Part1 conditioner.
That is what hides `models/` and `models_x128_backup`, whose DiTs use the 60-sim
`ParamConditioner`. Anything hidden is listed in the sidebar with the reason.

What the app refuses to claim:

- **`noair` has no cavity planes.** `Wing_cavity` and `Main_cavity` are not
  offered for a `noair` build-up — FDS writes 3 planes, not 5, and a generated
  cavity field there would be a fiction. Same rule as the training loader's mask.
- **Geometry cannot extrapolate.** It is an 8-way embedding over *observed* flag
  combinations; an absent build-up gets a warning banner, not a silent field.
- **The selected model is best-available, not significantly best** — the sidebar
  says so, with the margin.

`app_common_part1.py`, `app_fundiff_part1.py` and `run_app.ps1` are
**byte-identical** across the projects that hold them. Never hand-edit one copy;
edit and re-copy. Full app contract: `..\APPS.md`.

---

### Repository layout

Run logs, analysis records and before-state snapshots are grouped so the project
root holds only what you run:

```
<project>/
  README.md            this file
  *.py                 all modules and entry points — flat, at the root
  models_*/            checkpoints + per-run provenance JSON
  app_assets/          part1_materials.json, selected_model.json
  docs/                results records and analyses (PART1_RESULTS.md, analysis_*.md, ...)
  logs/                paired .log / .err.log run logs — the provenance of every number
  archive/             before-state snapshots of deliberate edits (*.pre-*, *.bak)
```

**Python stays at the project root, deliberately.** Every module imports flat
(`from config_part1 import ...`) and `config.py` / `config_part1.py` derive
`PROJECT_DIR`, `MODEL_DIR`, `OUTPUT_DIR` and `SLICE_DIR` from `__file__` — moving
them into a `src/` package would silently repoint model and slice paths, and
those files must stay byte-identical across all eleven surrogate projects for
`verify_parity_part1.py` to pass. New run logs still land at the root; move them
into `logs/` when you tidy.

`CLAUDE.md` is git-ignored: it is the working brief for agent sessions, not part
of the published artefact.

---

## Notes and caveats

- **Config is the source of truth** for every hyperparameter — `config.py` for
  the architecture and the FAE/DiT recipe, `config_part1.py` for the corpus.
- **Retrain variance.** Deltas inside **±0.02 R²** are reported inconclusive, not
  as an architecture win. The two FunDiff replicates differ by 0.0014.
- **Small-data caveat.** The conditional distribution is near-deterministic and
  Part1 gives roughly 600 (sim × plane) functions; the strong conditioning
  collapses the generator toward a deterministic map. The `FAE(recon)` −
  `DiT(gen)` gap measures how close Stage 2 gets to the Stage-1 ceiling.
- **The two splits are different experiments.** Always state whether a number
  came from `hash` or `system`.
- Weights are git-ignored by extension; the two selected checkpoints are negated
  back in so a fresh clone can predict. Run logs and per-run JSON stay tracked —
  they are the provenance of every number above.

## Related

`..\CLAUDE.md` (project map, Part1 contract) · `..\APPS.md` (app and deployment
contract) · `bs8414_fundiff_kan_surrogate` (the KAN-conditioning sibling) ·
`bs8414_physicsnemo_surrogate`, `bs8414_slice_surrogate`, `samba_MLP`
(deterministic slice-field arms on the same target).
