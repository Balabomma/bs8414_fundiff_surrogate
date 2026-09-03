"""Data layer for the BS 8414 Part1 geometry-variant corpus.

Parent: `data_loader.py` (60-sim corpus). Rewritten rather than extended because
four of its assumptions no longer hold on `D:\\Bs8414_05052026\\Part1\\_completed`:

  1. CHIDs carry HRR and mesh tokens          -> Part1 CHIDs carry geometry flags
  2. material MATL ids are a fixed short list -> Part1 names them per system
  3. properties are namelist constants        -> several are &RAMP tables
  4. every deck writes 24 thermocouples       -> 169 of 186 write 16

Everything it produces is keyed by NAME, never by column position, so a change of
column order in an FDS output cannot silently substitute the wrong channel.

Run `python data_loader_part1.py` for a full corpus audit: discovery counts,
per-system feature table, normalised-feature ranges and the split census.
"""
import glob
import hashlib
import os
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config_part1 import (
    SIMS_DIR, SPLIT_MODE, TRAIN_RATIO, VALID_RATIO,
    CLADDING_SYSTEMS, INSULATION_SYSTEMS, INSULATION_BY_MATL,
    GEOMETRY_FLAGS, GEOMETRY_BITS, GEOMETRY_NAMES,
    MATERIAL_FEATURES, N_MATERIAL_FEATURES, N_INPUT_PARAMS,
    N_TIMESTEPS, T_END, DT_DEVC, N_SENSORS, EXTERNAL_PREFIXES,
    HRR_CHANNELS, N_HRR_CHANNELS, EXCLUDED_CHIDS,
)

# ──────────────────────────────────────────────────────────────────────
# FDS namelist parsing
# ──────────────────────────────────────────────────────────────────────

def _namelists(text, kind):
    """Yield the raw text of every `&<kind> ... /` block.

    Scans for the terminating '/' with a quote-state machine instead of a regex:
    MATL blocks contain quoted ids ('PIR_Int_1') and RAMP names, and a
    non-greedy `.*?/` stops at the first slash inside a quoted string.
    """
    for m in re.finditer(rf"&{kind}\b", text):
        i, in_str = m.end(), False
        while i < len(text):
            c = text[i]
            if c == "'":
                in_str = not in_str
            elif c == "/" and not in_str:
                break
            i += 1
        yield text[m.start():i + 1]


def _str_field(block, key):
    """Read a quoted value, tolerating an FDS array index on the key.

    Solid-phase reaction fields are written indexed — SPEC_ID(1,1)='WOOD_VOLATILES',
    MATL_ID(1,1)='CEDAR_CHAR' — so a plain `key=` pattern matches none of them and
    returns None for every deck in the corpus.
    """
    m = re.search(rf"\b{key}(?:\([^)]*\))?\s*=\s*'([^']*)'", block)
    return m.group(1) if m else None


def _num_field(block, key):
    """Read a scalar, refusing to match a longer key that ends with `key`.

    Without the leading guard, CONDUCTIVITY would match CONDUCTIVITY_RAMP's
    value and SPECIFIC_HEAT would match SPECIFIC_HEAT_RAMP.
    """
    m = re.search(rf"(?<![A-Z_]){key}\s*=\s*(-?[\d.]+(?:[Ee][+-]?\d+)?)", block)
    return float(m.group(1)) if m else None


def parse_ramps(text):
    """Return {ramp_id: (T[], F[])} for every &RAMP in the deck, T ascending."""
    ramps = {}
    for b in _namelists(text, "RAMP"):
        rid = _str_field(b, "ID")
        t, f = _num_field(b, "T"), _num_field(b, "F")
        if rid is None or t is None or f is None:
            continue
        ramps.setdefault(rid, []).append((t, f))
    out = {}
    for rid, pts in ramps.items():
        pts.sort()
        out[rid] = (np.array([p[0] for p in pts]), np.array([p[1] for p in pts]))
    return out


def resolve_property(block, key, ramps, at_temperature=20.0):
    """Constant `key`, or its `key_RAMP` evaluated at `at_temperature`.

    FDS multiplies a *_RAMP by the base value when both are given; in this corpus
    the ramp always carries the absolute property (no base constant is present
    alongside it), so the ramp value is used directly. Evaluated at 20 degC -
    the initial-state value, which is the like-for-like counterpart of the
    constant-property materials it sits in a feature vector beside.
    """
    const = _num_field(block, key)
    if const is not None:
        return const
    ramp_id = _str_field(block, f"{key}_RAMP")
    if ramp_id and ramp_id in ramps:
        t, f = ramps[ramp_id]
        return float(np.interp(at_temperature, t, f))
    return 0.0


def _matl_blocks(text):
    return {_str_field(b, "ID"): b for b in _namelists(text, "MATL")
            if _str_field(b, "ID")}


def _surf_matls(text, surf_id):
    """MATL_ID entries bound to one SURF, in layer order."""
    for b in _namelists(text, "SURF"):
        if _str_field(b, "ID") == surf_id:
            return re.findall(r"MATL_ID\([^)]*\)\s*=\s*'([^']+)'", b)
    return []


def _matl_props(block, ramps):
    """The six MATL quantities the feature vector needs, ramps resolved."""
    if block is None:
        return {"cp": 0.0, "k": 0.0, "rho": 0.0, "hor": 0.0, "tref": 0.0,
                "reactive": 0.0, "spec": None}
    return {
        "cp": resolve_property(block, "SPECIFIC_HEAT", ramps),
        "k": resolve_property(block, "CONDUCTIVITY", ramps),
        "rho": resolve_property(block, "DENSITY", ramps),
        "hor": _num_field(block, "HEAT_OF_REACTION") or 0.0,
        "tref": _num_field(block, "REFERENCE_TEMPERATURE") or 0.0,
        "reactive": 1.0 if _num_field(block, "N_REACTIONS") else 0.0,
        # SPEC_ID(1,1)='WOOD_VOLATILES' — links the solid to its gas reaction
        "spec": _str_field(block, "SPEC_ID"),
    }


def extract_deck(fds_path):
    """Parse one deck into {core, ins, reac, ins_name}.

    Wiring contract, verified across all 186 decks:
        SURF 'ACM_Core'   -> core material = first non-Aluminium layer
                             (ACM build-ups are Al / core / Al; solid claddings
                             list the single material)
        SURF 'Insulation' -> the insulation product, one layer
        REAC whose FUEL == core's SPEC_ID(1,1) -> the core's combustion reaction;
                             falls back to ETHYLENE (the burner) for inert cores
    """
    with open(fds_path, "r", errors="replace") as f:
        text = f.read()

    ramps = parse_ramps(text)
    matls = _matl_blocks(text)

    core_ids = [m for m in _surf_matls(text, "ACM_Core") if m != "Aluminium"]
    core_id = core_ids[0] if core_ids else None
    ins_ids = _surf_matls(text, "Insulation")
    ins_id = ins_ids[0] if ins_ids else None

    core = _matl_props(matls.get(core_id), ramps)
    ins = _matl_props(matls.get(ins_id), ramps)

    reac = {"hoc": 0.0, "soot": 0.0}
    if core["spec"]:
        for b in _namelists(text, "REAC"):
            if _str_field(b, "FUEL") == core["spec"]:
                reac = {"hoc": _num_field(b, "HEAT_OF_COMBUSTION") or 0.0,
                        "soot": _num_field(b, "SOOT_YIELD") or 0.0}
                break

    return {"core": core, "ins": ins, "reac": reac,
            "core_matl": core_id, "ins_matl": ins_id,
            "ins_name": INSULATION_BY_MATL.get(ins_id)}


def material_vector(deck):
    """The 13 raw material features, in MATERIAL_FEATURES order."""
    c, i, r = deck["core"], deck["ins"], deck["reac"]
    return [c["cp"], c["k"], c["rho"], c["hor"], c["tref"], c["reactive"],
            i["cp"], i["k"], i["rho"], i["hor"], i["tref"],
            r["hoc"], r["soot"]]


# ──────────────────────────────────────────────────────────────────────
# Feature normalisation
# ──────────────────────────────────────────────────────────────────────
# Fixed physical bounds, not statistics of the corpus: no train/test leakage,
# and the encoding does not shift when the FDS batch adds cases.
#
# Conductivity and density are log-scaled. The Part1 corpus spans k from 0.02
# (phenolic foam) to 235 W/mK (aluminium 1050) and rho from 14 to 7850 kg/m3;
# on a linear scale every organic material collapses into the bottom 0.1% of the
# range and the encoder cannot separate cedar from plywood.
MATERIAL_NORM = {
    "core_specific_heat": ("lin", 0.0, 4.0),
    "core_conductivity": ("log", 0.01, 300.0),
    "core_density": ("log", 10.0, 8000.0),
    "core_heat_of_reaction": ("lin", 0.0, 2500.0),
    "core_ref_temperature": ("lin", 0.0, 600.0),
    "core_is_reactive": ("lin", 0.0, 1.0),
    "ins_specific_heat": ("lin", 0.0, 2.0),
    "ins_conductivity": ("log", 0.01, 300.0),
    "ins_density": ("log", 10.0, 8000.0),
    "ins_heat_of_reaction": ("lin", 0.0, 500.0),
    "ins_ref_temperature": ("lin", 0.0, 600.0),
    "reac_heat_of_combustion": ("lin", 0.0, 5.0e4),
    "reac_soot_yield": ("lin", 0.0, 0.2),
}


def normalize_material(raw):
    """Map the 13 raw features into [0, 1]; clipped, so a new material outside
    the declared bounds saturates instead of producing an outlier."""
    out = []
    for val, name in zip(raw, MATERIAL_FEATURES):
        mode, lo, hi = MATERIAL_NORM[name]
        if mode == "log":
            v = (np.log10(max(val, lo)) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
        else:
            v = (val - lo) / (hi - lo) if hi > lo else 0.0
        out.append(float(np.clip(v, 0.0, 1.0)))
    return out


# ──────────────────────────────────────────────────────────────────────
# CHID parsing
# ──────────────────────────────────────────────────────────────────────

def parse_chid(chid):
    """`BS8414_ACM_A2_MW_noair_nocb` -> ('ACM_A2', 'MW', 5, 'ACM_A2_MW').

    Returns (cladding, insulation_or_None, geom_id, base_system). Insulation is
    None for the DCLG reference decks, whose names carry no insulation suffix;
    the caller fills it from the deck's SURF 'Insulation' material.
    """
    name = chid[len("BS8414_"):] if chid.startswith("BS8414_") else chid

    geom_id = 0
    for flag in GEOMETRY_FLAGS:
        if re.search(rf"_{flag}(?=_|$)", name):
            geom_id |= GEOMETRY_BITS[flag]
            name = re.sub(rf"_{flag}(?=_|$)", "", name)

    base = name
    cladding = insulation = None
    # Longest cladding key first: 'ACM_A2' must win over any shorter prefix.
    for clad in sorted(CLADDING_SYSTEMS, key=len, reverse=True):
        if base == clad:
            cladding = clad
            break
        if base.startswith(clad + "_"):
            suffix = base[len(clad) + 1:]
            if suffix in INSULATION_SYSTEMS:
                cladding, insulation = clad, suffix
                break
    return cladding, insulation, geom_id, base


# ──────────────────────────────────────────────────────────────────────
# Output-file loading
# ──────────────────────────────────────────────────────────────────────

def _read_fds_csv(path):
    """FDS CSV: row 1 is units, row 2 is the header. Returns (time, df)."""
    df = pd.read_csv(path, skiprows=1)
    df.columns = [str(c).strip() for c in df.columns]
    return df.iloc[:, 0].to_numpy(dtype=np.float64), df


def _on_time_grid(time, values):
    """Resample onto the canonical 0..1800 s / 10 s grid.

    Truncates a longer run (DCLG_Test5 goes to 2000 s) and pads a short one
    (HPL_WC_noair_nogap stops one step early), returning the validity mask so a
    padded tail is never scored as if it were data.
    """
    n = min(len(time), N_TIMESTEPS)
    out = np.zeros((N_TIMESTEPS, values.shape[1]), dtype=np.float32)
    mask = np.zeros(N_TIMESTEPS, dtype=np.float32)
    out[:n] = values[:n]
    mask[:n] = 1.0
    if n < N_TIMESTEPS:  # hold the last observed value under a zeroed mask
        out[n:] = values[n - 1]
    return out, mask


def load_thermocouples(devc_csv):
    """The 16 external channels, selected by name, on the canonical grid."""
    time, df = _read_fds_csv(devc_csv)
    cols = [c for c in df.columns[1:] if c.startswith(EXTERNAL_PREFIXES)]
    if len(cols) != N_SENSORS:
        raise ValueError(f"{os.path.basename(devc_csv)}: matched {len(cols)} "
                         f"external channels, expected {N_SENSORS}")
    values, mask = _on_time_grid(time, df[cols].to_numpy(dtype=np.float32))
    return values, mask, cols


def load_hrr(hrr_csv):
    """The HRR_CHANNELS energy-budget columns, by name, on the canonical grid."""
    time, df = _read_fds_csv(hrr_csv)
    missing = [c for c in HRR_CHANNELS if c not in df.columns]
    if missing:
        raise ValueError(f"{os.path.basename(hrr_csv)}: missing {missing}")
    values, mask = _on_time_grid(time, df[HRR_CHANNELS].to_numpy(dtype=np.float32))
    return values, mask


# ──────────────────────────────────────────────────────────────────────
# Discovery
# ──────────────────────────────────────────────────────────────────────

def discover_simulations(sims_dir=None, verbose=True):
    """Scan the corpus. Every rejection is counted and reported, never silent."""
    sims_dir = sims_dir or SIMS_DIR
    sims, rejected = [], []

    if not os.path.isdir(sims_dir):
        raise FileNotFoundError(f"Part1 corpus not found: {sims_dir}")

    for folder in sorted(os.listdir(sims_dir)):
        path = os.path.join(sims_dir, folder)
        if not os.path.isdir(path) or folder.startswith((".", "_")):
            continue

        if folder in EXCLUDED_CHIDS:
            rejected.append((folder, f"excluded: {EXCLUDED_CHIDS[folder]}"))
            continue

        cladding, insulation, geom_id, base = parse_chid(folder)
        if cladding is None:
            rejected.append((folder, "CHID does not parse to a known cladding"))
            continue

        devc = glob.glob(os.path.join(path, "*_devc.csv"))
        hrr = glob.glob(os.path.join(path, "*_hrr.csv"))
        fds = glob.glob(os.path.join(path, "*.fds"))
        if not devc:
            rejected.append((folder, "no _devc.csv"))
            continue
        if not hrr:
            rejected.append((folder, "no _hrr.csv"))
            continue
        if not fds:
            rejected.append((folder, "no .fds deck"))
            continue

        sims.append({"chid": folder, "folder": path, "devc_csv": devc[0],
                     "hrr_csv": hrr[0], "fds_file": fds[0], "cladding": cladding,
                     "insulation": insulation, "geom_id": geom_id, "base": base})

    if verbose:
        print(f"  corpus: {sims_dir}")
        print(f"  {len(sims)} usable, {len(rejected)} rejected")
        for chid, why in rejected:
            print(f"    - {chid}: {why}")
    return sims


def build_dataset(sims_dir=None, verbose=True):
    """Assemble (params, tc, hrr, mask, meta, sensor_names).

    params  (N, 16)               [cladding_id, insulation_id, geom_id, 13 mat]
    tc      (N, 181, 16)          external thermocouples, degC
    hrr     (N, 181, 5)           HRR_CHANNELS, kW
    mask    (N, 181)              1 where the simulation actually reported
    """
    sims = discover_simulations(sims_dir, verbose=verbose)

    params, tc_out, hrr_out, masks, meta = [], [], [], [], []
    sensor_names = None
    failures = []

    for sim in sims:
        try:
            deck = extract_deck(sim["fds_file"])
            insulation = sim["insulation"] or deck["ins_name"]
            if insulation not in INSULATION_SYSTEMS:
                raise ValueError(
                    f"insulation unresolved (SURF 'Insulation' -> "
                    f"{deck['ins_matl']!r})")

            tc, tc_mask, cols = load_thermocouples(sim["devc_csv"])
            hrr, hrr_mask = load_hrr(sim["hrr_csv"])
        except Exception as exc:
            failures.append((sim["chid"], str(exc)))
            continue

        if sensor_names is None:
            sensor_names = cols
        elif cols != sensor_names:
            failures.append((sim["chid"], "thermocouple channel order differs "
                                          "from the first deck read"))
            continue

        raw = material_vector(deck)
        params.append([CLADDING_SYSTEMS[sim["cladding"]],
                       INSULATION_SYSTEMS[insulation],
                       sim["geom_id"]] + normalize_material(raw))
        tc_out.append(tc)
        hrr_out.append(hrr)
        # A step counts only where both output files reported it.
        masks.append(np.minimum(tc_mask, hrr_mask))
        meta.append({"chid": sim["chid"], "folder": sim["folder"],
                     "cladding": sim["cladding"], "insulation": insulation,
                     "geom_id": sim["geom_id"],
                     "geometry": GEOMETRY_NAMES[sim["geom_id"]],
                     "base": sim["base"], "core_matl": deck["core_matl"],
                     "ins_matl": deck["ins_matl"], "material_raw": raw,
                     "n_valid": int(np.minimum(tc_mask, hrr_mask).sum())})

    if failures and verbose:
        print(f"  {len(failures)} failed to load:")
        for chid, why in failures:
            print(f"    - {chid}: {why}")

    params = np.asarray(params, dtype=np.float32)
    tc_out = np.asarray(tc_out, dtype=np.float32)
    hrr_out = np.asarray(hrr_out, dtype=np.float32)
    masks = np.asarray(masks, dtype=np.float32)

    if verbose:
        print(f"  params {params.shape}  tc {tc_out.shape}  hrr {hrr_out.shape}")
        partial = [m["chid"] for m in meta if m["n_valid"] < N_TIMESTEPS]
        for chid in partial:
            n = next(m["n_valid"] for m in meta if m["chid"] == chid)
            print(f"  partial: {chid} ({n}/{N_TIMESTEPS} steps, padded+masked)")

    return params, tc_out, hrr_out, masks, meta, sensor_names


# ──────────────────────────────────────────────────────────────────────
# Splitting
# ──────────────────────────────────────────────────────────────────────

def _bucket(key):
    """Stable float in [0, 1) from a string — same function as the 60-sim
    pipeline, so split behaviour is unchanged where the key is unchanged."""
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def assign_split(meta_entry, mode=None, train_ratio=TRAIN_RATIO,
                 valid_ratio=VALID_RATIO):
    """train/valid/test for one simulation.

    mode="hash"   bucket on the CHID: geometry variants of a system scatter
    mode="system" bucket on the cladding+insulation base: all 8 geometry
                  variants of a system land in the same split, so test systems
                  are unseen build-ups
    """
    mode = mode or SPLIT_MODE
    key = meta_entry["chid"] if mode == "hash" else \
        f"{meta_entry['cladding']}_{meta_entry['insulation']}"
    b = _bucket(key)
    if b < train_ratio:
        return "train"
    if b < train_ratio + valid_ratio:
        return "valid"
    return "test"


def split_indices(meta, mode=None):
    idx = {"train": [], "valid": [], "test": []}
    for i, m in enumerate(meta):
        idx[assign_split(m, mode)].append(i)
    return {k: np.asarray(v, dtype=int) for k, v in idx.items()}


# ──────────────────────────────────────────────────────────────────────
# Scaling and Dataset
# ──────────────────────────────────────────────────────────────────────

class ChannelScaler:
    """Per-channel standardisation fitted on TRAIN ROWS ONLY.

    The 60-sim pipeline fitted its output scaler on the full array before
    splitting. That is a (mild) leak of test statistics into training, and the
    project's own rules call it out, so it is not carried over here. Masked
    (padded) timesteps are excluded from the fit.
    """

    def __init__(self):
        self.mean = None
        self.scale = None

    def fit(self, values, mask):
        flat = values.reshape(-1, values.shape[-1])
        keep = mask.reshape(-1) > 0
        flat = flat[keep]
        self.mean = flat.mean(axis=0)
        self.scale = flat.std(axis=0)
        self.scale[self.scale < 1e-6] = 1.0
        return self

    def transform(self, values):
        return (values - self.mean) / self.scale

    def inverse(self, values):
        if torch.is_tensor(values):
            mean = torch.as_tensor(self.mean, dtype=values.dtype, device=values.device)
            scale = torch.as_tensor(self.scale, dtype=values.dtype, device=values.device)
            return values * scale + mean
        return values * self.scale + self.mean

    def state_dict(self):
        return {"mean": np.asarray(self.mean), "scale": np.asarray(self.scale)}

    def load_state_dict(self, state):
        self.mean = np.asarray(state["mean"])
        self.scale = np.asarray(state["scale"])
        return self


class Part1Dataset(Dataset):
    """(params, tc, hrr, mask, time) — both targets in standardised space."""

    def __init__(self, params, tc, hrr, mask, time_array):
        self.params = torch.as_tensor(params, dtype=torch.float32)
        self.tc = torch.as_tensor(tc, dtype=torch.float32)
        self.hrr = torch.as_tensor(hrr, dtype=torch.float32)
        self.mask = torch.as_tensor(mask, dtype=torch.float32)
        self.time_array = torch.as_tensor(time_array, dtype=torch.float32)

    def __len__(self):
        return len(self.params)

    def __getitem__(self, i):
        return (self.params[i], self.tc[i], self.hrr[i], self.mask[i],
                self.time_array)


def prepare_data_splits(params, tc, hrr, mask, meta, mode=None, verbose=True):
    """Split, fit scalers on train only, and wrap in datasets."""
    time_array = np.linspace(0.0, 1.0, N_TIMESTEPS).astype(np.float32)
    idx = split_indices(meta, mode)

    if len(idx["train"]) == 0:
        raise RuntimeError("empty training split")

    tc_scaler = ChannelScaler().fit(tc[idx["train"]], mask[idx["train"]])
    hrr_scaler = ChannelScaler().fit(hrr[idx["train"]], mask[idx["train"]])

    tc_s = tc_scaler.transform(tc).astype(np.float32)
    hrr_s = hrr_scaler.transform(hrr).astype(np.float32)

    datasets = {}
    for name, ids in idx.items():
        datasets[name] = (Part1Dataset(params[ids], tc_s[ids], hrr_s[ids],
                                       mask[ids], time_array)
                          if len(ids) else None)

    info = {"mode": mode or SPLIT_MODE, "indices": idx,
            "meta": {k: [meta[i] for i in v] for k, v in idx.items()}}

    if verbose:
        print(f"\n  split mode: {info['mode']}")
        for name in ("train", "valid", "test"):
            entries = info["meta"][name]
            print(f"    {name:<6} {len(entries):>3} sims  "
                  f"{len({e['base'] for e in entries}):>2} systems  "
                  f"{len({e['geom_id'] for e in entries})}/8 geometries")

    return datasets, tc_scaler, hrr_scaler, info, time_array


# ──────────────────────────────────────────────────────────────────────
# Corpus audit
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 78)
    print("  BS 8414 Part1 geometry-variant corpus — audit")
    print("=" * 78)

    params, tc, hrr, mask, meta, sensor_names = build_dataset()

    print(f"\n  thermocouples ({len(sensor_names)}):")
    for n in sensor_names:
        print(f"    {n}")
    print(f"\n  HRR channels ({N_HRR_CHANNELS}): {', '.join(HRR_CHANNELS)}")

    print("\n  per-system material features (raw, one row per base system):")
    print(f"    {'system':<18}{'core':<16}{'ins':<18}"
          f"{'cp':>6}{'k':>9}{'rho':>8}{'HoR':>8}{'Tref':>7}{'rx':>4}{'HoC':>9}")
    seen = set()
    for m in sorted(meta, key=lambda x: x["base"]):
        if m["base"] in seen:
            continue
        seen.add(m["base"])
        r = m["material_raw"]
        print(f"    {m['base']:<18}{str(m['core_matl']):<16}"
              f"{str(m['ins_matl']):<18}{r[0]:>6.2f}{r[1]:>9.3f}{r[2]:>8.0f}"
              f"{r[3]:>8.1f}{r[4]:>7.0f}{r[5]:>4.0f}{r[11]:>9.0f}")

    print("\n  normalised feature ranges (must sit inside [0, 1] without "
          "piling on a bound):")
    for j, name in enumerate(MATERIAL_FEATURES):
        col = params[:, 3 + j]
        print(f"    {name:<26} min={col.min():.3f}  max={col.max():.3f}  "
              f"mean={col.mean():.3f}  unique={len(np.unique(col))}")

    print("\n  geometry census:")
    for gid in range(8):
        n = sum(1 for m in meta if m["geom_id"] == gid)
        print(f"    {gid}  {GEOMETRY_NAMES[gid]:<20} {n:>3} sims")

    print("\n  cladding census:")
    for clad in CLADDING_SYSTEMS:
        n = sum(1 for m in meta if m["cladding"] == clad)
        print(f"    {clad:<18} {n:>3} sims")

    print("\n  insulation census:")
    for ins in INSULATION_SYSTEMS:
        n = sum(1 for m in meta if m["insulation"] == ins)
        print(f"    {ins:<18} {n:>3} sims")

    print(f"\n  thermocouple range: {tc.min():.1f} .. {tc.max():.1f} degC")
    print(f"  HRR range:          {hrr[:, :, 0].min():.1f} .. "
          f"{hrr[:, :, 0].max():.1f} kW")

    for mode in ("hash", "system"):
        prepare_data_splits(params, tc, hrr, mask, meta, mode=mode)
