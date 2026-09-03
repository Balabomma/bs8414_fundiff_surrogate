"""Configuration for the BS8414 slice-field surrogate — FunDiff variant.

Implements FunDiff (Wang, Dou, Shan, Liu & Lu, "FunDiff: diffusion models over
function spaces for physics-informed generative modeling", Nat. Commun. 17:5749,
2026, doi:10.1038/s41467-026-72292-0) adapted to the BS 8414-1 facade-fire
surrogate task.

Design decisions (locked with the user, 2026-07-20):
  * FULL FunDiff, STAGED — Stage 1 trains a Function Autoencoder (FAE) that is
    usable standalone as a deterministic comparison entry; Stage 2 trains a
    latent rectified-flow Diffusion Transformer (DiT) on the frozen FAE latents,
    conditioned on the 16-d parameter vector + slice id.
  * SPATIOTEMPORAL 3D functions — one "function" is the full temperature field
    T(t, y, z) for a single (simulation, slice).  The CViT-style decoder is
    queried at continuous (t, y, z) coordinates, giving arbitrary temporal AND
    spatial super-resolution.  Physics priors (ambient floor, growth-phase
    monotonicity) are enforced *inside the FAE decoder / on sampled coords*,
    exactly as the paper enforces divergence-free / periodic constraints in the
    FAE decoder — decoupled from the diffusion stage.

Target and split are IDENTICAL to the other slice surrogates so the comparison
tooling lines up: 5 planes x 128x64 x 181 timesteps, deterministic 70/15/15
hash split (= 42/9/9 on the full 60-sim set).

Data is REUSED (read-only) from the already-extracted PhysicsNeMo project to
avoid a costly re-extraction — the same pattern the PhysicsNeMo / Samba projects
already use to read the KAN / baseline training set.  Repoint SHARED_DATA_DIR /
NORM_STATS_PATH to re-extract locally if ever needed.
"""
import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
# PART1_MODEL_DIR keeps a Part1 run from overwriting the 60-sim checkpoints:
# train_fae.py and train_dit.py write fixed names (fae_best.pt, dit_best.pt)
# into MODEL_DIR, and models/ already holds the 60-sim ones. Silent
# overwrites have already cost this family three retrains.
MODEL_DIR = os.path.join(PROJECT_DIR,
                         os.environ.get("PART1_MODEL_DIR", "models"))

# ── Shared (read-only) data from the PhysicsNeMo project ─────────────
SHARED_DATA_DIR = r"D:\VS_projects\bs8414_physicsnemo_surrogate\data"
NUMPY_DIR = os.path.join(SHARED_DATA_DIR, "numpy")
METADATA_PATH = os.path.join(SHARED_DATA_DIR, "metadata.json")
NORM_STATS_PATH = r"D:\VS_projects\bs8414_physicsnemo_surrogate\models\norm_stats.json"

# ── Cladding systems / design space (shared schema) ──────────────────
CLADDING_SYSTEMS = {
    "Test_1_PE_PIR": 0, "Test_3_FRPE_PIR": 1,
    "Test_5_LCM_PIR": 2, "Test_7_FRPE_Phenolic": 3,
}
HRR_LEVELS = [1333, 1667, 2000, 2100, 2333]
MESH_SIZES = {"M008": 0.08, "M009": 0.09, "M010": 0.10}

MATERIAL_FEATURES = [
    "core_specific_heat", "core_conductivity", "core_density",
    "core_heat_of_reaction", "core_ref_temperature", "core_is_reactive",
    "ins_specific_heat", "ins_conductivity", "ins_density",
    "ins_heat_of_reaction", "ins_ref_temperature",
    "reac_heat_of_combustion", "reac_soot_yield",
]
N_MATERIAL_FEATURES = len(MATERIAL_FEATURES)
N_INPUT_PARAMS = 3 + N_MATERIAL_FEATURES  # 16  (cladding_id, hrr_norm, mesh_norm + 13)

# ── Slices ───────────────────────────────────────────────────────────
SLICE_IDS = [
    "Main_external", "Wing_cavity", "Wing_external", "Main_cavity", "Mid_section",
]
N_SLICES = len(SLICE_IDS)
SLICE_ID_MAP = {name: idx for idx, name in enumerate(SLICE_IDS)}

# ── Spatiotemporal grid ──────────────────────────────────────────────
FIELD_HEIGHT = 128   # Z / vertical
FIELD_WIDTH = 64     # horizontal
T_END = 1800.0
DT_SLCF = 10.0
N_TIMESTEPS = 181

# Encoder patchify pads T up to a multiple of PATCH_T; the decoder still queries
# the true 181x128x64 grid, so padding never touches the reconstruction target.
GRID_SHAPE = (N_TIMESTEPS, FIELD_HEIGHT, FIELD_WIDTH)

# ── True physical geometry of the slices (from fdsreader slc.extent) ─
# The shared extraction was FIXED on 2026-07-22 to transpose native fdsreader
# slices (in-plane horizontal x Z) before resizing, so stored data is now
# **axis0 (FIELD_HEIGHT=128) = vertical Z (10 m)** and **axis1 (FIELD_WIDTH=64)
# = in-plane horizontal (4 m)** — up-sampling both axes from native (e.g.
# 51x126 -> 128x64) instead of down-sampling the info-rich vertical.  Verified
# by burner location (hot source pinned at low Z = low axis0).
# (Pre-fix data had axis0=horizontal; see numpy_x128_old / models_x128_backup.)
DOMAIN_EXTENT = {"x": (0.0, 4.0), "y": (0.0, 4.0), "z": (0.0, 10.0)}  # metres
STORED_AXIS0_IS_HORIZONTAL = False  # axis0(128)=Z vertical, axis1(64)=in-plane horizontal

# Per-slice: cutting plane, its position on the normal axis (m), the in-plane
# horizontal axis + its range (m), and the vertical range (m).  Horizontal is X
# for the XZ (Main) planes and Y for the YZ (Wing/Mid) planes; vertical is Z.
SLICE_GEOMETRY = {
    "Main_external": {"plane": "XZ", "normal": "y", "pos": 1.77,
                      "horiz_axis": "x", "horiz_range": (0.0, 4.0), "vert_range": (0.0, 10.0)},
    "Main_cavity":   {"plane": "XZ", "normal": "y", "pos": 1.87,
                      "horiz_axis": "x", "horiz_range": (0.0, 4.0), "vert_range": (0.0, 10.0)},
    "Wing_cavity":   {"plane": "YZ", "normal": "x", "pos": 0.53,
                      "horiz_axis": "y", "horiz_range": (0.0, 4.0), "vert_range": (0.0, 10.0)},
    "Wing_external": {"plane": "YZ", "normal": "x", "pos": 0.62,
                      "horiz_axis": "y", "horiz_range": (0.0, 4.0), "vert_range": (0.0, 10.0)},
    "Mid_section":   {"plane": "YZ", "normal": "x", "pos": 1.78,
                      "horiz_axis": "y", "horiz_range": (0.0, 4.0), "vert_range": (0.0, 10.0)},
}

# ── Ambient (for the physics floor) ──────────────────────────────────
T_AMBIENT = 18.0  # deg C

# ── Deterministic split (identical hash scheme) ──────────────────────
TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15

DEVICE = "cuda"

# ═════════════════════════════════════════════════════════════════════
#  Function Autoencoder (FAE) — Stage 1
# ═════════════════════════════════════════════════════════════════════
# ViT-style 3D patchify -> Perceiver latent bottleneck -> self-attention
# transformer stack (encoder); CViT continuous cross-attention (decoder).
EMBED_DIM = 256          # token / latent width (paper: 256)
MLP_WIDTH = 512          # transformer MLP width (paper: 512)
N_HEADS = 8              # attention heads (paper: 8)
ENC_DEPTH = 8            # encoder self-attention blocks (paper: 8-layer encoder)
DEC_DEPTH = 4            # decoder cross-attention blocks (paper: 4-layer decoder)
N_LATENT_TOKENS = 128    # Perceiver latent queries (compact latent = N x EMBED_DIM)

PATCH_T = 8              # temporal patch (T padded 181 -> 184 -> multiple of 8 = 184? -> pad to 184)
PATCH_H = 16             # height patch  (128 / 16 = 8)
PATCH_W = 16             # width patch   (64 / 16 = 4)

# Coordinate embedding for the decoder queries (random Fourier features).
COORD_FOURIER_FEATURES = 64   # -> 2 * this * 3 raw features, projected to EMBED_DIM
COORD_FOURIER_SCALE = 8.0

FAE_DROPOUT = 0.0

# ── FAE training ─────────────────────────────────────────────────────
FAE_LR = 1e-3
FAE_WARMUP_STEPS = 300        # linear warm-up 0 -> LR  (paper used 2000 STEPS, but
                              # with ~16 steps/epoch on 125 functions that is ~125
                              # epochs = 31% of the budget; 300 steps ~= 20 epochs)
FAE_DECAY_EVERY = 2000        # exponential decay every N steps (paper)
FAE_DECAY_RATE = 0.9          # decay factor (paper)
FAE_WEIGHT_DECAY = 1e-5       # AdamW weight decay (paper)
FAE_BATCH_SIZE = 8            # functions per batch (paper used 16; 8 fits the 4090 with 181-frame fields)
FAE_N_QUERY = 4096            # random (t,y,z) query points per function per step (paper)
FAE_EPOCHS = 400
FAE_PATIENCE = 60
FAE_EMA_DECAY = 0.999

# Resolution-invariance augmentation: randomly subsample the encoder input grid
# in time by one of these factors (paper subsamples by {1,2,4,8}).
FAE_ENC_SUBSAMPLE_T = [1, 2, 4]

# ── FAE reconstruction loss weights ──────────────────────────────────
LAMBDA_SSIM = 0.15            # SSIM on decoded full frames (sampled timesteps)
LAMBDA_GROWTH = 1e-2          # growth-phase temporal monotonicity (soft, on coord pairs)
LAMBDA_ENERGY = 5e-3          # mean-T <-> HRR correlation
SSIM_EVAL_FRAMES = 4          # decode this many full frames/function for the SSIM term

# Hard physics floor in the decoder: output = ambient + softplus(raw) so the
# reconstructed field can never drop below ambient (analogue of the paper's
# hard divergence-free / periodic decoder modifications).
USE_AMBIENT_FLOOR = True

# Growth phase is the fraction of the run over which T is expected non-decreasing.
GROWTH_PHASE_FRAC = 0.6       # t in [0, 0.6] treated as growth for the monotonicity prior

# ═════════════════════════════════════════════════════════════════════
#  Latent Diffusion Transformer (DiT) — Stage 2
# ═════════════════════════════════════════════════════════════════════
DIT_DEPTH = 8                 # DiT transformer blocks (paper: 8)
DIT_EMBED_DIM = 256           # (paper: 256)
DIT_MLP_WIDTH = 512           # (paper: 512)
DIT_HEADS = 8                 # (paper: 8)
DIT_DROPOUT = 0.0

# Conditioning: the 16-d parameter vector + slice-id embedding -> cond token.
COND_EMBED_DIM = 256
SLICE_EMBED_DIM = 32

# ── DiT training (rectified flow) ────────────────────────────────────
DIT_LR = 1e-3
DIT_WARMUP_STEPS = 2000
DIT_DECAY_EVERY = 2000
DIT_DECAY_RATE = 0.9
DIT_WEIGHT_DECAY = 1e-5
DIT_BATCH_SIZE = 32           # latent vectors are tiny -> large batch is cheap
DIT_EPOCHS = 3000
DIT_PATIENCE = 300
DIT_EMA_DECAY = 0.9995

# ── Inference / sampling ─────────────────────────────────────────────
# Rectified flow: integrate dz/dt = v(z,t) from t=0 (noise) to t=1 (data).
N_SAMPLING_STEPS = 50         # Euler / RK ODE steps (paper: 20-50)
N_COND_SAMPLES = 8            # samples averaged for the deterministic conditional-mean readout
INFERENCE_SEED = 1234         # fixed seed so the readout is reproducible

# ── Generative uncertainty quantification (evaluate_uq.py) ───────────
# FunDiff is generative: N independent latent samples -> N decoded fields give a
# per-pixel predictive mean and std (a principled uncertainty map, no dropout /
# ensemble needed).  Chavare (arXiv:2607.18294) calibration metrics in uq_metrics.py.
UQ_N_SAMPLES = 24             # DiT samples per test case for the mean/std estimate

# ── Chunked full-grid decode (evaluation) ────────────────────────────
# The full 181*128*64 = 1.48M query grid is decoded in chunks to bound memory.
DECODE_CHUNK = 65536
