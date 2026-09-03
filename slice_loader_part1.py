"""Slice-field data layer for the BS 8414 Part1 geometry-variant corpus.

Consumes the `.npz` files written by `extract_slices_part1.py` and produces the
same conditioning vector as the thermocouple models — `config_part1.py` and
`data_loader_part1.py` are shared verbatim, so a slice model and a thermocouple
model see byte-identically the same parameters, the same geometry ids and the
same train/valid/test assignment.

The one thing this layer adds over the thermocouple loader is the **plane
presence mask**. Two of the five planes only exist where there is a ventilated
cavity, so 92 of 185 cases carry 3 planes. Both the loss and every metric must
skip absent planes rather than score zeros — that is what `plane_mask` is for,
and it is returned from every path here so it cannot be forgotten downstream.

    python slice_loader_part1.py            # corpus audit

Fields are held in RAM as float16 (~2.8 GB for the full corpus) and converted to
float32 per batch. Decompressing a .npz per sample instead would dominate epoch
time at this corpus size.
"""
import glob
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from config_part1 import (
    SLICE_DIR, SLICE_IDS, SLICE_ID_MAP, SLICE_REQUIRES_CAVITY, N_SLICES,
    COL_GEOM,
    FIELD_HEIGHT, FIELD_WIDTH, N_TIMESTEPS, T_AMBIENT,
    GEOMETRY_NAMES, GEOM_HAS_CAVITY, N_INPUT_PARAMS,
    CLADDING_SYSTEMS, INSULATION_SYSTEMS, SPLIT_MODE,
)
from data_loader_part1 import assign_split, load_hrr
from config_part1 import SIMS_DIR, HRR_CHANNELS, EXCLUDED_CHIDS

CLADDING_BY_ID = {v: k for k, v in CLADDING_SYSTEMS.items()}
INSULATION_BY_ID = {v: k for k, v in INSULATION_SYSTEMS.items()}

HRR_COL = HRR_CHANNELS.index("HRR")


def load_hrr_curves(meta, sims_dir=None, verbose=True):
    """Total HRR(t) in kW for each case, read from the source `_hrr.csv`.

    The 60-sim energy-magnitude constraint correlated mean predicted temperature
    against `params[:, 1]`, the normalised burner HRR. In Part1 that column is
    the insulation id and the burner HRR is identical in all 186 decks, so the
    constraint has nothing to correlate against and would silently regress onto
    an arbitrary categorical code.

    The physically meaningful replacement is the case's ACTUAL total heat release,
    which does vary — it is exactly the cladding/insulation combustion
    contribution the geometry study is about. Read here rather than baked into the
    .npz so the extraction does not have to be re-run to add it.
    """
    sims_dir = sims_dir or SIMS_DIR
    curves = np.zeros((len(meta), N_TIMESTEPS), dtype=np.float32)
    missing = []
    for i, m in enumerate(meta):
        path = os.path.join(sims_dir, m["chid"], f"{m['chid']}_hrr.csv")
        if not os.path.isfile(path):
            missing.append(m["chid"])
            continue
        values, _ = load_hrr(path)
        curves[i] = values[:, HRR_COL]
    if verbose and missing:
        print(f"  {len(missing)} cases without an _hrr.csv: "
              f"{', '.join(missing[:5])}{' ...' if len(missing) > 5 else ''}")
    return curves


def load_corpus(slice_dir=None, max_sims=None, verbose=True):
    """Read every extracted case into memory.

    Returns (params, fields, time_mask, plane_mask, meta):
        params      (N, 16)                       float32
        fields      (N, 5, 181, 128, 64)          float16, degC
        time_mask   (N, 5, 181)                   1 where that plane reported
        plane_mask  (N, 5)                        1 where the plane exists at all
    """
    slice_dir = slice_dir or SLICE_DIR
    paths = sorted(glob.glob(os.path.join(slice_dir, "*.npz")))

    # Honour EXCLUDED_CHIDS here rather than relying on the extractor having
    # skipped the case. The .npz files are extracted once and cached for ~50
    # min, so a case excluded AFTER extraction still has its file on disk and
    # would silently re-enter the slice corpus while the thermocouple pipeline
    # dropped it - the two would then disagree on n and on the split, which is
    # exactly what the parity check exists to prevent.
    kept = []
    for path in paths:
        chid = os.path.splitext(os.path.basename(path))[0]
        if chid in EXCLUDED_CHIDS:
            if verbose:
                print(f"  excluded: {chid} ({EXCLUDED_CHIDS[chid]})")
            continue
        kept.append(path)
    paths = kept

    if not paths:
        raise FileNotFoundError(
            f"no extracted slices in {slice_dir}. "
            f"Run: python extract_slices_part1.py")
    if max_sims:
        paths = paths[:max_sims]

    params, fields, time_mask, plane_mask, meta = [], [], [], [], []
    inconsistent = []

    for i, path in enumerate(paths):
        with np.load(path, allow_pickle=True) as f:
            present = np.asarray(f["plane_present"], dtype=bool)
            geom_id = int(f["geom_id"])

            # A plane that exists but should not (or the reverse) means the deck
            # and the CHID disagree about the geometry — report, do not silently
            # fold it into training.
            expected = np.array([GEOM_HAS_CAVITY[geom_id]
                                 or not SLICE_REQUIRES_CAVITY[s]
                                 for s in SLICE_IDS])
            if not np.array_equal(present, expected):
                inconsistent.append(
                    (str(f["chid"]), GEOMETRY_NAMES[geom_id],
                     [SLICE_IDS[j] for j in range(N_SLICES)
                      if present[j] != expected[j]]))

            params.append(np.asarray(f["params"], dtype=np.float32))
            fields.append(np.asarray(f["fields"], dtype=np.float16))
            time_mask.append(np.asarray(f["time_mask"], dtype=np.float32))
            plane_mask.append(present.astype(np.float32))
            meta.append({
                "chid": str(f["chid"]),
                "cladding": str(f["cladding"]),
                "insulation": str(f["insulation"]),
                "geom_id": geom_id,
                "geometry": GEOMETRY_NAMES[geom_id],
                "planes": [SLICE_IDS[j] for j in range(N_SLICES) if present[j]],
            })

        if verbose and (i + 1) % 25 == 0:
            print(f"    loaded {i + 1}/{len(paths)}")

    params = np.stack(params)
    fields = np.stack(fields)
    time_mask = np.stack(time_mask)
    plane_mask = np.stack(plane_mask)

    if verbose:
        gb = fields.nbytes / 1024 ** 3
        print(f"  {len(meta)} cases, fields {fields.shape} float16 ({gb:.2f} GB)")
        if inconsistent:
            print(f"  {len(inconsistent)} cases where present planes disagree "
                  f"with the CHID geometry:")
            for chid, geom, planes in inconsistent:
                print(f"    - {chid} ({geom}): {', '.join(planes)}")

    return params, fields, time_mask, plane_mask, meta


class FieldScaler:
    """Single global standardisation of the temperature field.

    One statistic for all five planes, not per-plane: the planes share a physical
    scale and per-plane statistics would make a cavity plane's degC error
    incomparable with an external plane's. Fitted on TRAIN rows only, over
    timesteps and planes that actually exist.
    """

    def __init__(self):
        self.mean = None
        self.scale = None

    def fit(self, fields, time_mask, plane_mask):
        total = count = 0.0
        sq = 0.0
        for i in range(fields.shape[0]):
            for p in range(N_SLICES):
                if plane_mask[i, p] == 0:
                    continue
                steps = time_mask[i, p] > 0
                if not steps.any():
                    continue
                block = fields[i, p][steps].astype(np.float32)
                total += block.sum()
                sq += (block.astype(np.float64) ** 2).sum()
                count += block.size
        self.mean = float(total / count)
        var = max(float(sq / count) - self.mean ** 2, 1e-6)
        self.scale = float(np.sqrt(var))
        return self

    def transform(self, values):
        return (values - self.mean) / self.scale

    def inverse(self, values):
        return values * self.scale + self.mean

    def state_dict(self):
        return {"mean": self.mean, "scale": self.scale}

    def load_state_dict(self, state):
        self.mean = float(state["mean"])
        self.scale = float(state["scale"])
        return self


class Part1SliceDataset(Dataset):
    """(params, fields, time_mask, plane_mask, time_array).

    Fields are standardised on the fly so the corpus stays float16 in RAM.
    """

    def __init__(self, params, fields, time_mask, plane_mask, scaler, time_array,
                 hrr_curves=None):
        self.params = torch.as_tensor(params, dtype=torch.float32)
        self.fields = fields          # float16 numpy view, not a tensor copy
        self.time_mask = torch.as_tensor(time_mask, dtype=torch.float32)
        self.plane_mask = torch.as_tensor(plane_mask, dtype=torch.float32)
        self.scaler = scaler
        self.time_array = torch.as_tensor(time_array, dtype=torch.float32)
        self.hrr_curves = (torch.as_tensor(hrr_curves, dtype=torch.float32)
                           if hrr_curves is not None
                           else torch.zeros(len(self.params), N_TIMESTEPS))

    def __len__(self):
        return len(self.params)

    def __getitem__(self, i):
        field = torch.from_numpy(
            self.scaler.transform(self.fields[i].astype(np.float32)))
        return (self.params[i], field, self.time_mask[i], self.plane_mask[i],
                self.time_array, self.hrr_curves[i])


def prepare_slice_splits(params, fields, time_mask, plane_mask, meta,
                         mode=None, verbose=True, hrr_curves=None):
    """Split by the shared Part1 rule and fit the scaler on train only."""
    time_array = np.linspace(0.0, 1.0, N_TIMESTEPS).astype(np.float32)

    idx = {"train": [], "valid": [], "test": []}
    for i, m in enumerate(meta):
        idx[assign_split(m, mode)].append(i)
    idx = {k: np.asarray(v, dtype=int) for k, v in idx.items()}

    if len(idx["train"]) == 0:
        raise RuntimeError("empty training split")

    scaler = FieldScaler().fit(fields[idx["train"]], time_mask[idx["train"]],
                               plane_mask[idx["train"]])

    datasets = {}
    for name, ids in idx.items():
        datasets[name] = (Part1SliceDataset(
            params[ids], fields[ids], time_mask[ids], plane_mask[ids],
            scaler, time_array,
            hrr_curves[ids] if hrr_curves is not None else None)
            if len(ids) else None)

    info = {"mode": mode or SPLIT_MODE, "indices": idx,
            "meta": {k: [meta[i] for i in v] for k, v in idx.items()}}

    if verbose:
        print(f"\n  split mode: {info['mode']}   "
              f"field mean {scaler.mean:.1f} degC, sd {scaler.scale:.1f}")
        for name in ("train", "valid", "test"):
            entries = info["meta"][name]
            cavity = sum(1 for e in entries if GEOM_HAS_CAVITY[e["geom_id"]])
            print(f"    {name:<6} {len(entries):>3} sims  "
                  f"{cavity:>3} with cavity (5 planes)  "
                  f"{len(entries) - cavity:>3} without (3 planes)")

    return datasets, scaler, info, time_array


# ──────────────────────────────────────────────────────────────────────
# Plane existence — which SLCFs a geometry actually has
# ──────────────────────────────────────────────────────────────────────
# FDS writes 5 slice planes for a case with a ventilated cavity and 3 for a
# `noair` case, because two of the planes are *inside* the cavity and there is
# nothing there to slice. The loader has always carried this as `plane_mask`,
# and the trainer draws (case, plane) samples only where it is set, so no model
# is ever trained on an absent plane.
#
# What was missing is the INFERENCE side. Nothing stopped a caller asking a
# trained model for `Wing_cavity` on a `noair` design, and nothing in the model
# refuses: it returns a full, finite, confident 128x64 temperature field for a
# surface that does not exist. Demonstrated directly - a `noair` case and a
# baseline case queried for the same cavity plane both return complete fields.
#
# These helpers make the physical contract explicit at the point of use. They
# deliberately do NOT zero the output: a zero-filled cavity plane is exactly the
# imputation this pipeline refuses, because it teaches that removing the cavity
# makes the cavity cold. An impossible query returns NaN - undefined, not cold.

def plane_exists(geom_ids, slice_ids):
    """(B,) bool: does this (geometry, plane) pair exist in FDS?

    Accepts torch tensors or numpy arrays of integer ids.
    """
    g = np.asarray(torch.as_tensor(geom_ids).cpu().numpy()
                   if torch.is_tensor(geom_ids) else geom_ids, dtype=int)
    s = np.asarray(torch.as_tensor(slice_ids).cpu().numpy()
                   if torch.is_tensor(slice_ids) else slice_ids, dtype=int)
    needs_cavity = np.array([SLICE_REQUIRES_CAVITY[SLICE_IDS[i]] for i in s])
    has_cavity = np.array([GEOM_HAS_CAVITY[int(i)] for i in g])
    return ~needs_cavity | has_cavity


def existing_planes(geom_id):
    """Slice ids a given geometry actually has: 5 with a cavity, 3 without."""
    return [i for i, name in enumerate(SLICE_IDS)
            if GEOM_HAS_CAVITY[int(geom_id)] or not SLICE_REQUIRES_CAVITY[name]]


def mask_impossible(pred, geom_ids, slice_ids):
    """Set predictions for non-existent (geometry, plane) pairs to NaN."""
    ok = torch.as_tensor(plane_exists(geom_ids, slice_ids), device=pred.device)
    out = pred.clone()
    out[~ok] = float("nan")
    return out


class GuardedSliceModel(torch.nn.Module):
    """Wrap any Part1 slice model so it cannot answer for a plane that does not
    exist.

    The six slice/diffusion projects differ only in `model_part1.py` - that file
    IS the variable under test - so the plane-existence contract is enforced here
    instead of being pasted into each of them. Wrapping also keeps the guard on
    whatever architecture is loaded, including ones added later.

    Training is left untouched: the sampler already draws (case, plane) pairs
    only where `plane_mask` is set, so no impossible pair ever reaches the loss,
    and NaN-ing during training would risk poisoning gradients if it ever did.
    The guard therefore applies in eval mode only, which is exactly where the
    hole was - a caller asking a trained model for `Wing_cavity` on a `noair`
    design used to get a full, finite, confident field back.
    """

    def __init__(self, model, scaler=None, t_ambient=T_AMBIENT):
        super().__init__()
        self.model = model
        # Ambient floor expressed in the model's own standardised space, the
        # same construction `set_output_scaling` uses on the thermocouple side.
        # Unlike the plane guard this applies in TRAINING TOO, so the model
        # does not spend capacity on sub-ambient predictions - that is what made
        # all four slice architectures fail the sub-ambient gate on ~45% of
        # points.
        #
        # Caveat, measured rather than assumed: unlike the thermocouple truth
        # (minimum 17.88 degC), the slice TRUTH does reach -11.81 degC on 0.34%
        # of its points. The clamp therefore biases that small minority upward.
        # It is still a large net win (pooled field R2 +0.03, SSIM 0.19 -> 0.88
        # on every architecture), but it is a favourable approximation, not an
        # exact physical bound. See evaluate_slices_part1.py for the numbers.
        if scaler is not None:
            amb = (float(t_ambient) - float(scaler.mean)) / float(scaler.scale)
            self.register_buffer("ambient_scaled", torch.tensor(amb))
        else:
            self.register_buffer("ambient_scaled", None)

    def forward(self, params, slice_ids, time_window, *a, **kw):
        out = self.model(params, slice_ids, time_window, *a, **kw)
        if self.ambient_scaled is not None:
            out = torch.maximum(out, self.ambient_scaled.to(out.dtype))
        if self.training:
            return out
        geom = params[:, COL_GEOM].long()
        return mask_impossible(out, geom, slice_ids)

    def __getattr__(self, name):
        # transparently expose the wrapped model's own attributes
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


@torch.no_grad()
def predict_case(model, params_row, time_window, scaler=None):
    """Predict every plane that EXISTS for one case - 5 or 3, not always 5.

    params_row  (16,) or (1, 16)
    returns     {plane_name: (T, 1, H, W)} containing only real planes, so the
                caller cannot accidentally consume a cavity field for a design
                that has no cavity.
    """
    p = params_row.reshape(1, -1)
    geom = int(p[0, COL_GEOM].item())
    ids = existing_planes(geom)
    out = {}
    for sid in ids:
        field = model(p, torch.tensor([sid], device=p.device),
                      time_window.reshape(1, -1))
        f = field[0]
        if scaler is not None:
            f = f * float(scaler.scale) + float(scaler.mean)
        out[SLICE_IDS[sid]] = f
    return out


def masked_field_loss_weights(plane_mask, slice_weights):
    """Per-plane loss weights zeroed where the plane does not exist.

    Renormalised per sample so a 3-plane case is not implicitly down-weighted
    against a 5-plane one — otherwise every `noair` geometry contributes ~60% of
    a baseline case's gradient purely because of what FDS wrote to disk.
    """
    w = torch.as_tensor(slice_weights, dtype=plane_mask.dtype,
                        device=plane_mask.device)
    w = w.unsqueeze(0) * plane_mask
    return w / w.sum(dim=1, keepdim=True).clamp(min=1e-6)


if __name__ == "__main__":
    from config_part1 import SLICE_LOSS_WEIGHTS

    print("=" * 78)
    print("  BS 8414 Part1 slice corpus — audit")
    print("=" * 78)

    params, fields, time_mask, plane_mask, meta = load_corpus()

    print("\n  plane availability:")
    for j, sid in enumerate(SLICE_IDS):
        n = int(plane_mask[:, j].sum())
        tag = " (cavity only)" if SLICE_REQUIRES_CAVITY[sid] else ""
        print(f"    {sid:<16} {n:>3}/{len(meta)} cases{tag}")

    print("\n  geometry census:")
    for gid in range(8):
        sel = [m for m in meta if m["geom_id"] == gid]
        if sel:
            print(f"    {GEOMETRY_NAMES[gid]:<20} {len(sel):>3} cases, "
                  f"{len(sel[0]['planes'])} planes")

    valid = plane_mask.sum(axis=1) > 0
    sample = fields[:, 0][time_mask[:, 0] > 0].astype(np.float32)
    print(f"\n  Main_external field range: {sample.min():.1f} .. "
          f"{sample.max():.1f} degC")

    hrr_curves = load_hrr_curves(meta)
    peak = hrr_curves.max(axis=1)
    print(f"\n  peak HRR across cases: {peak.min():.0f} .. {peak.max():.0f} kW "
          f"(mean {peak.mean():.0f}) — this is what the energy constraint keys on")

    datasets, scaler, info, _ = prepare_slice_splits(
        params, fields, time_mask, plane_mask, meta, hrr_curves=hrr_curves)

    p, f, tm, pm, ta, hc = datasets["train"][0]
    print(f"\n  sample: params {tuple(p.shape)}  fields {tuple(f.shape)}  "
          f"time_mask {tuple(tm.shape)}  plane_mask {pm.tolist()}")
    w = masked_field_loss_weights(pm.unsqueeze(0), SLICE_LOSS_WEIGHTS)
    print(f"  loss weights for that sample: "
          f"{[round(x, 3) for x in w[0].tolist()]}  (sum {w.sum():.3f})")
