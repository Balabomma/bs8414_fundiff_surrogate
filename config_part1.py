"""Dataset contract for the BS 8414 **Part1 geometry-variant** corpus.

Parent: `config.py` (the 60-sim cladding x HRR x mesh corpus).
What changed, and why every number below differs from that file:

    data root   D:\\Bs8414_05052026\\Part1\\_completed   (186 completed sims)

The Part1 batch is a *geometry* sensitivity study, not an HRR/mesh study, so the
design space rotated:

  dropped   HRR  - every deck is HRRPUA=2333.3 kW/m2 (verified over all 186)
  dropped   mesh - every deck is dx=0.10 m, 16 meshes, T_END=1800 s
  added     geometry - 3 binary construction modifiers -> 8 combinations
  widened   cladding  - 4 DCLG systems -> 12 (8 generic + 4 DCLG references)
  added     insulation as an explicit axis - 5 products

The prediction target also shrank. 169 of the 186 decks instrument only the
external face; the Insulation Level 2 group survives in the 17 legacy `DCLG_*`
decks alone. Training the insulation head on 9% of the corpus would be a fiction,
so the Part1 target is the **16 external thermocouples** and two grouped decoders
(External LV1, External LV2). This mirrors `MLP_SENSORS=external` in the MLP
ablation project, and it costs the BR 135 *internal* criterion, which cannot be
assessed from external channels. The *external* criterion is unaffected.

Second target added at user request: the **`_hrr.csv` global energy budget**. The
burner ramp is identical in every deck, so the run-to-run variation in HRR is the
cladding/insulation combustion contribution - the quantity that drives the
thermocouple response in the first place. Only channels present in all 186 files
are used (the per-fuel `MLR_*` columns vary with the reaction set and are not).

Nothing in this file is read by the existing 60-sim pipeline. `config.py`,
`data_loader.py` and every `models*/` checkpoint are untouched.
"""
import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── data root ─────────────────────────────────────────────────────────────
# The FDS batch writes completed cases here; the loader rescans on every run,
# so cases finishing later are picked up without touching this file.
SIMS_DIR = os.environ.get(
    "PART1_SIMS_DIR", r"D:\Bs8414_05052026\Part1\_completed")

OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
MODEL_DIR = os.path.join(PROJECT_DIR, "models_part1")

# ── split ─────────────────────────────────────────────────────────────────
# "hash"   deterministic MD5-of-CHID bucketing, same function as the 60-sim
#          pipeline. Geometry variants of one system scatter across splits, so
#          the model sees every base system in training -> measures how well
#          geometry effects are learned, NOT system generalisation.
# "system" groups all 8 geometry variants of a cladding+insulation system into
#          the same split, so test systems are genuinely unseen. Harder and the
#          honest number for a "can it predict a new build-up" claim.
# Report which one a result came from; they are not comparable.
SPLIT_MODE = os.environ.get("PART1_SPLIT", "hash").lower()
if SPLIT_MODE not in ("hash", "system"):
    raise ValueError(f"PART1_SPLIT={SPLIT_MODE!r} unknown; expected hash|system")

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15

# ── design space ──────────────────────────────────────────────────────────
# Cladding id is taken from the CHID prefix, not from the core MATL name:
# `Core_MATERIAL` is reused by ACM-PE, DCLG_Test1_adv, DCLG_Test3 and DCLG_Test5
# with four different property sets, so the MATL id alone is not a system id.
CLADDING_SYSTEMS = {
    "ACM_A2": 0,          # A2 mineral-filled aluminium composite
    "ACM_PE": 1,          # polyethylene-cored aluminium composite
    "AL": 2,              # solid aluminium sheet
    "BRK": 3,             # red clay brick slip
    "CDR": 4,             # western red cedar
    "HPL": 5,             # high-pressure laminate
    "OSB": 6,             # oriented strand board
    "PLY": 7,             # plywood
    "DCLG_Test1_adv": 8,  # DCLG reference tests, carried for continuity with
    "DCLG_Test3": 9,      # the 60-sim corpus and the NUREG-1824 validation work
    "DCLG_Test5": 10,
    "DCLG_Test7": 11,
}
N_CLADDING = len(CLADDING_SYSTEMS)

# Insulation is resolved from the MATL bound to SURF ID='Insulation', which is
# unambiguous, rather than from the CHID suffix (the DCLG names carry no suffix).
INSULATION_SYSTEMS = {"MW": 0, "MWBC": 1, "PF": 2, "PIR": 3, "WC": 4}
N_INSULATION = len(INSULATION_SYSTEMS)

INSULATION_BY_MATL = {
    "MW_RWA45": "MW",            # stone wool slab, 47 kg/m3
    "MW_BEAMCLAD": "MWBC",       # beam-clad stone wool, 162 kg/m3
    "PF": "PF",                  # phenolic foam
    "Phenolic_Foam": "PF",       # DCLG Test 7 phenolic
    "Insulation_VIRGIN": "PIR",  # PIR with the 7-step intermediate scheme
    "STEEL": "WC",               # warm-cavity build-up: steel liner, no slab
}

# ── geometry axis ─────────────────────────────────────────────────────────
# Bit flags, so geom_id is reproducible from the CHID and reads back cleanly:
#   bit 0 (1) noair - ventilated air cavity removed, cladding sits on insulation
#   bit 1 (2) nogap - open joints between cladding panels closed
#   bit 2 (4) nocb  - cavity barriers removed
# geom_id 0 is the fully-featured baseline (cavity + gaps + barriers present).
GEOMETRY_FLAGS = ("noair", "nogap", "nocb")
GEOMETRY_BITS = {"noair": 1, "nogap": 2, "nocb": 4}
N_GEOMETRY = 8
GEOMETRY_NAMES = {
    0: "baseline", 1: "noair", 2: "nogap", 3: "noair_nogap",
    4: "nocb", 5: "noair_nocb", 6: "nogap_nocb", 7: "noair_nogap_nocb",
}

# `noair` decks have no cavity, so their FDS writes 3 slice planes instead of 5.
# Irrelevant to the sensor models; the slice pipeline must mask on it.
GEOM_HAS_CAVITY = {gid: not (gid & GEOMETRY_BITS["noair"]) for gid in range(8)}

# ── material features ─────────────────────────────────────────────────────
# Same 13 quantities as the 60-sim corpus so feature semantics stay comparable.
# Extraction differs: Part1 decks name materials per system (ACM_A2_CORE,
# CEDAR, RED_CLAY_BRICK, ...) and several give CONDUCTIVITY/SPECIFIC_HEAT as a
# &RAMP rather than a constant, which the old fixed-name extractor read as 0.
MATERIAL_FEATURES = [
    "core_specific_heat", "core_conductivity", "core_density",
    "core_heat_of_reaction", "core_ref_temperature", "core_is_reactive",
    "ins_specific_heat", "ins_conductivity", "ins_density",
    "ins_heat_of_reaction", "ins_ref_temperature",
    "reac_heat_of_combustion", "reac_soot_yield",
]
N_MATERIAL_FEATURES = len(MATERIAL_FEATURES)

# params vector: [cladding_id, insulation_id, geom_id] + 13 material features
N_INPUT_PARAMS = 3 + N_MATERIAL_FEATURES  # 16
COL_CLADDING, COL_INSULATION, COL_GEOM = 0, 1, 2

# ── time grid ─────────────────────────────────────────────────────────────
T_END = 1800.0
DT_DEVC = 10.0
N_TIMESTEPS = 181
T_AMBIENT = 18.0  # &MISC TMPA=18.0 in every Part1 deck

# ── thermocouple target ───────────────────────────────────────────────────
EXTERNAL_PREFIXES = ("External_LV1", "External_LV2")
N_SENSORS = 16
SENSOR_GROUPS = {
    "External_LV1": 8,  # main x5 + wing x3, indices 0-7
    "External_LV2": 8,  # main x5 + wing x3, indices 8-15
}
GROUP_SIZES = [8, 8]

KEY_THERMOCOUPLES = {
    "External_LV2_main02(1029)": "TC 1029 - External LV2 Main",
    "External_LV1_main02(1003)": "TC 1003 - External LV1 Main",
}

# ── HRR target ────────────────────────────────────────────────────────────
# Present in all 186 `_hrr.csv` files. Q_PRES / Q_PART / Q_DIFF / MLR_AIR /
# MLR_PRODUCTS are in the common set too but sit at numerical-noise magnitude
# for these cases, so they are not made prediction targets. Per-fuel MLR columns
# (MLR_WOOD_VOLATILES, MLR_PF_FUEL, MLR_Gas_1..7, ...) vary with the reaction
# set - 24 distinct header layouts across the corpus - and cannot be a fixed
# target vector at all.
HRR_CHANNELS = ["HRR", "Q_RADI", "Q_CONV", "Q_COND", "Q_TOTAL"]
N_HRR_CHANNELS = len(HRR_CHANNELS)
HRR_NONNEGATIVE = ("HRR",)  # clamped at 0 in the model; the Q_* terms sign-flip

# Relative weight of the HRR head against the thermocouple head in the loss.
# Both are trained in standardised space, so this is a plain balance term.
LAMBDA_HRR = float(os.environ.get("PART1_LAMBDA_HRR", "0.3"))

# ── slice-field target ────────────────────────────────────────────────────
# Used by the slice surrogates (bs8414_slice_surrogate, bs8414_physicsnemo_
# surrogate, samba_MLP, bs8414_samba_mlp_surrogate, bs8414_fundiff_surrogate).
# Ignored by the thermocouple models, which never import these names.
#
# The two cavity planes DO NOT EXIST in the 92 `noair` decks — with no ventilated
# cavity there is nothing to slice, so FDS writes 3 SLCFs instead of 5. This is
# physically correct, not missing data, and it must not be imputed: a zero-filled
# cavity plane would teach the model that removing the cavity makes the cavity
# cold. SLICE_REQUIRES_CAVITY marks the affected planes; the loader emits a
# per-plane presence mask and the loss skips absent planes entirely.
SLICE_IDS = [
    "Main_external",   # PBY=1.4      always present
    "Wing_cavity",     # PBX=0.525    cavity only
    "Wing_external",   # PBX=0.6      always present
    "Main_cavity",     # PBY=1.475    cavity only
    "Mid_section",     # PBX=1.822 (1.61572 in the DCLG_Test3 decks)
]
N_SLICES = len(SLICE_IDS)
SLICE_ID_MAP = {name: i for i, name in enumerate(SLICE_IDS)}
SLICE_REQUIRES_CAVITY = {"Wing_cavity": True, "Main_cavity": True,
                         "Main_external": False, "Wing_external": False,
                         "Mid_section": False}

# axis0 = Z (true vertical), axis1 = in-plane horizontal. The 2026-07 geometry
# fix: extraction transposes the native fdsreader (horizontal, Z) layout so the
# information-rich 10 m vertical is up-sampled rather than crushed to 64.
FIELD_HEIGHT = 128
FIELD_WIDTH = 64

# One compressed .npz per simulation rather than 181 x 5 loose .npy files:
# the loose layout would put ~168,000 files on disk for this corpus. Fields are
# stored float16 — at 1000 degC the representable spacing is ~0.5 degC, far
# inside LES run-to-run scatter, and it halves a 5.6 GB corpus.
# ONE shared location, not a per-project copy. Extraction takes ~45 min and
# produces ~2.8 GB; every slice project reads the same files, exactly as the
# samba and physicsnemo projects already share the 60-sim data root.
SLICE_DIR = os.environ.get(
    "PART1_SLICE_DIR",
    os.path.join(os.path.dirname(PROJECT_DIR), "bs8414_slice_surrogate",
                 "data", "part1_slices"))
SLICE_DTYPE = "float16"

# Order matches SLICE_IDS. Wing planes carry the flame-spread signal and stay
# up-weighted, as in the 60-sim recipe.
SLICE_LOSS_WEIGHTS = [1.0, 2.0, 2.0, 1.0, 1.3]

# Slice training hyperparameters are declared HERE rather than imported from
# each project's `config.py`, because two of the five slice projects have none
# to import from (`bs8414_samba_mlp_surrogate` is Hydra-YAML, and
# `bs8414_fundiff_surrogate` has no plain `train.py`). Declaring them once is
# what lets `train_slices_part1.py` be byte-identical across all five — the same
# contract the thermocouple projects use.
# Values carried from bs8414_slice_surrogate/config.py, which is where the
# batch-4 + accum-4 setting was tuned against the 24 GB 4090's memory ceiling.
SLICE_BATCH_SIZE = 4
SLICE_ACCUM_STEPS = 4          # effective batch 16 at batch-4 memory
SLICE_NUM_EPOCHS = 300
SLICE_PATIENCE = 50
SLICE_LEARNING_RATE = 3e-4
SLICE_WEIGHT_DECAY = 1e-3
SLICE_TEMPORAL_WINDOW = 16
SLICE_EMA_DECAY = 0.999
SLICE_USE_AMP = True

LAMBDA_SSIM = 0.15
LAMBDA_SMOOTH = 5e-4
LAMBDA_GROWTH = 1e-2           # T non-decreasing through the growth phase
LAMBDA_ENERGY = 5e-3           # mean field T tracks the case's measured HRR

# ── excluded cases ────────────────────────────────────────────────────────
# Excluded by name, with the reason, rather than silently filtered by a shape
# check - a case that drops out later should be a visible decision, not a gap.
EXCLUDED_CHIDS = {
    "BS8414_DCLG_Test1_adv_debug":
        "debug case; Core_MATERIAL HEAT_OF_REACTION=1500 vs 2300 in every "
        "other DCLG_Test1_adv deck, so it is a different system under the "
        "same name",
    "BS8414_DCLG_Test7":
        "declares insulation system PF like 33 other decks, but its "
        "Phenolic_Foam MATL is inert - HEAT_OF_REACTION absent (0 kJ/kg) with "
        "REFERENCE_TEMPERATURE=429 degC - where every other PF deck decomposes "
        "at HEAT_OF_REACTION=400 with no explicit reference temperature "
        "(cp 1.2 vs 1.5, k 0.020 vs 0.022, rho 32 vs 39). Same defect class as "
        "the debug case above: a different system under the same name. Found "
        "2026-08-19 by explain_part1.check_group_determinism, which verifies "
        "that an insulation id fixes its material block; this was the only "
        "violation in 185 cases",
}

# `BS8414_DCLG_Test5` runs to T_END=2000 s (201 rows) - truncated to the common
# 181-step grid by the loader rather than excluded.
# `BS8414_HPL_WC_noair_nogap` stops at 180 rows - padded and masked.

# ── model / training ──────────────────────────────────────────────────────
GEOMETRY_EMBED_DIM = 16
CLADDING_EMBED_DIM = 12
INSULATION_EMBED_DIM = 8

LSTM_HIDDEN_SIZE = 96
LSTM_NUM_LAYERS = 2
ATTENTION_HEADS = 4
EMBEDDING_DIM = 24
DROPOUT = 0.08
NUM_KNOTS = 8

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 16
NUM_EPOCHS = 500
PATIENCE = 60

DEVICE = "cuda"
