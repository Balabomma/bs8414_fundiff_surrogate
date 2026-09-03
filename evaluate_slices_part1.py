"""Frozen evaluation contract for the Part1 slice-field surrogate.

Scores full 181-step rollouts, not the strided training windows, so the number
reported is the one a user of the surrogate would actually get.

    python evaluate_slices_part1.py --model-dir models_part1

Reports, per split:
  * pooled R2 / RMSE / SSIM over the temperature field, in degC
  * **per plane** — the two cavity planes only exist for 93 of 185 cases and are
    the hardest; a pooled number that hides a cavity failure is not an answer
  * **per geometry** — the point of this corpus
  * physics gates, pass/fail

Absent planes are excluded everywhere by `plane_mask`. They are never scored as
zeros, and the per-plane rows state how many cases each average covers.
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from config_part1 import (
    DEVICE, N_SLICES, SLICE_IDS, N_TIMESTEPS, T_AMBIENT, GEOMETRY_NAMES,
    DT_DEVC, PROJECT_DIR, FIELD_HEIGHT, FIELD_WIDTH,
)
from slice_loader_part1 import (
    load_corpus, load_hrr_curves, prepare_slice_splits, FieldScaler,
)
from slice_losses_part1 import ssim_loss
from model_part1 import Part1Surrogate, MODEL_NAME

DEFAULT_MODEL_DIR = os.path.join(PROJECT_DIR, "models_part1")
GROWTH_END_STEP = int(720.0 / DT_DEVC)
ROLLOUT_CHUNK = 16  # timesteps per forward pass; the decoder is memory-hungry


def load_ensemble(model_dir, device):
    paths = sorted(glob.glob(os.path.join(model_dir, "*member*.pt")))
    if not paths:
        raise SystemExit(f"no checkpoints in {model_dir}")
    models, ckpt = [], None
    for path in paths:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = Part1Surrogate().to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        models.append(model)
    scaler = FieldScaler().load_state_dict(ckpt["field_scaler"])
    return models, scaler, ckpt


@torch.no_grad()
def rollout(models, params, plane, time_array, device):
    """Full 181-step field for one (case, plane), averaged over the ensemble.

    No horizontal-flip TTA. It is not applicable to this architecture and the
    parent's implementation is a no-op: `validate.ensemble_predict` computes
    `p = model(...).flip(-1)` and then averages `0.5 * (out + out_flip.flip(-1))`,
    where the second flip undoes the first, so the two terms are identical and
    the result equals the plain prediction (dropout is off under eval()).

    The deeper reason it cannot work here: TTA needs an input transform whose
    effect on the output is known. This decoder takes no spatial input — it
    generates the field from (params, plane, time) — so the only available
    "augmentation" is flipping the OUTPUT, and averaging a field with its own
    mirror image just imposes exact horizontal symmetry. The BS 8414 rig is not
    horizontally symmetric (the wing sits on one side of the main face), so that
    would be a physics error, not a variance reduction.
    """
    out = torch.zeros(N_TIMESTEPS, FIELD_HEIGHT, FIELD_WIDTH, device=device)

    for start in range(0, N_TIMESTEPS, ROLLOUT_CHUNK):
        sl = slice(start, min(start + ROLLOUT_CHUNK, N_TIMESTEPS))
        times = time_array[sl].unsqueeze(0).to(device)
        p = params.unsqueeze(0).to(device)
        s = plane.view(1).to(device)

        acc = None
        for model in models:
            pred = model(p, s, times)[0, :, 0]
            acc = pred if acc is None else acc + pred
        out[sl] = acc / len(models)
    return out


def field_metrics(pred, true, name=""):
    """R2, RMSE and SSIM over a stack of fields, all in degC."""
    ss_res = float(((true - pred) ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(((true - pred) ** 2).mean()))

    p = torch.from_numpy(pred).unsqueeze(1).float()
    t = torch.from_numpy(true).unsqueeze(1).float()
    lo, hi = float(t.min()), float(t.max())
    span = max(hi - lo, 1e-6)
    ssim = 1.0 - float(ssim_loss((p - lo) / span, (t - lo) / span))
    return {"r2": r2, "rmse": rmse, "ssim": ssim}


def evaluate_split(name, models, dataset, split_meta, scaler, device):
    if dataset is None or len(dataset) == 0:
        print(f"\n  [{name}] empty split")
        return None

    per_plane = {sid: {"pred": [], "true": [], "cases": 0} for sid in SLICE_IDS}
    per_geom = {}
    all_pred, all_true = [], []

    for i in range(len(dataset)):
        params, field, time_mask, plane_mask, time_array, _ = dataset[i]
        geom = split_meta[i]["geometry"]

        for plane in range(N_SLICES):
            if plane_mask[plane] == 0:
                continue
            steps = time_mask[plane].numpy() > 0
            pred = rollout(models, params, torch.tensor(plane), time_array,
                           device).cpu().numpy()
            pred = scaler.inverse(pred)[steps]
            true = scaler.inverse(field[plane].numpy())[steps]

            sid = SLICE_IDS[plane]
            per_plane[sid]["pred"].append(pred)
            per_plane[sid]["true"].append(true)
            per_plane[sid]["cases"] += 1
            per_geom.setdefault(geom, {"pred": [], "true": []})
            per_geom[geom]["pred"].append(pred)
            per_geom[geom]["true"].append(true)
            all_pred.append(pred)
            all_true.append(true)

    pooled = field_metrics(np.concatenate(all_pred), np.concatenate(all_true))
    print(f"\n  [{name}]  {len(dataset)} cases, "
          f"{sum(v['cases'] for v in per_plane.values())} (case, plane) rollouts")
    print(f"    pooled field    R2 {pooled['r2']:>7.4f}   "
          f"RMSE {pooled['rmse']:>7.2f} degC   SSIM {pooled['ssim']:.4f}")

    print(f"    per plane")
    plane_rows = {}
    for sid in SLICE_IDS:
        entry = per_plane[sid]
        if not entry["cases"]:
            print(f"      {sid:<16} absent in every case of this split")
            plane_rows[sid] = None
            continue
        m = field_metrics(np.concatenate(entry["pred"]),
                          np.concatenate(entry["true"]))
        print(f"      {sid:<16} n={entry['cases']:>3}  R2 {m['r2']:>7.4f}   "
              f"RMSE {m['rmse']:>7.2f} degC   SSIM {m['ssim']:.4f}")
        plane_rows[sid] = dict(m, cases=entry["cases"])

    print(f"    per geometry")
    geom_rows = {}
    for geom in [GEOMETRY_NAMES[g] for g in range(8) if GEOMETRY_NAMES[g] in per_geom]:
        m = field_metrics(np.concatenate(per_geom[geom]["pred"]),
                          np.concatenate(per_geom[geom]["true"]))
        n_cases = sum(1 for e in split_meta if e["geometry"] == geom)
        print(f"      {geom:<20} n={n_cases:>2}  R2 {m['r2']:>7.4f}   "
              f"RMSE {m['rmse']:>7.2f} degC   SSIM {m['ssim']:.4f}")
        geom_rows[geom] = dict(m, cases=n_cases)

    stacked = np.stack([p for p in all_pred if p.shape[0] == N_TIMESTEPS]) \
        if any(p.shape[0] == N_TIMESTEPS for p in all_pred) else None
    gates = {}
    if stacked is not None:
        sub_ambient = int((stacked < T_AMBIENT - 0.5).sum())
        growth = stacked[:, :GROWTH_END_STEP + 1]
        worst = float(np.diff(growth, axis=1).mean(axis=(2, 3)).min())
        gates = {
            "sub_ambient_points": sub_ambient,
            "sub_ambient_pass": sub_ambient == 0,
            "worst_mean_growth_drop_degC": round(worst, 3),
            "growth_monotonic_pass": worst > -5.0,
        }
        print(f"    physics")
        print(f"      sub-ambient points        {sub_ambient:>8}   "
              f"{'PASS' if gates['sub_ambient_pass'] else 'FAIL'}")
        print(f"      worst mean growth drop    {worst:>8.3f} degC   "
              f"{'PASS' if gates['growth_monotonic_pass'] else 'FAIL'}")

    return {"n_cases": len(dataset), "pooled": pooled, "planes": plane_rows,
            "geometry": geom_rows, "physics": gates}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--split", default=None, choices=("hash", "system"))
    ap.add_argument("--max-sims", type=int, default=None)
    ap.add_argument("--splits", default="valid,test",
                    help="which splits to score (train is slow and rarely needed)")
    args = ap.parse_args()

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    models, scaler, ckpt = load_ensemble(args.model_dir, device)
    split_mode = args.split or ckpt.get("split_mode")

    print("=" * 78)
    print(f"  Part1 slice surrogate evaluation — {args.model_dir}")
    print(f"  {MODEL_NAME}, {len(models)} member(s), split {split_mode}, "
          f"TTA off (see rollout() — the parent's flip TTA is a no-op)")
    print("=" * 78)

    params, fields, time_mask, plane_mask, meta = load_corpus(
        max_sims=args.max_sims, verbose=False)
    hrr_curves = load_hrr_curves(meta, verbose=False)
    datasets, _, info, _ = prepare_slice_splits(
        params, fields, time_mask, plane_mask, meta, mode=split_mode,
        verbose=False, hrr_curves=hrr_curves)

    results = {}
    for name in [s.strip() for s in args.splits.split(",")]:
        results[name] = evaluate_split(name, models, datasets[name],
                                       info["meta"][name], scaler, device)

    if results.get("valid") and results.get("test"):
        combined = 0.5 * (results["valid"]["pooled"]["r2"]
                          + results["test"]["pooled"]["r2"])
        print(f"\n  combined valid+test field R2: {combined:.4f}")
        results["combined_r2"] = combined

    out = os.path.join(args.model_dir, "evaluation_slices_part1.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\n  written: {out}")


if __name__ == "__main__":
    main()
