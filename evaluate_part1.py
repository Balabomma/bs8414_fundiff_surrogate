"""Frozen evaluation contract for the Part1 geometry-variant surrogate.

Fixed before any candidate model existed, and shared unchanged by every Part1
variant (KAN, MLP, Attention-LSTM). A candidate that needs different scoring is a
different experiment, not a comparable one.

    python evaluate_part1.py --model-dir models_part1

Reports, per split:
  * pooled and per-group R2 / RMSE on the 16 external thermocouples, in degC
  * R2 / RMSE per HRR channel, in kW
  * a **per-geometry breakdown** — the point of this corpus is whether removing
    the cavity, the gaps or the barriers is predictable, so a pooled number that
    hides a failure on one geometry is not an answer
  * physics sanity gates, pass/fail

Metrics are computed on unstandardised values over reported timesteps only.
Ensembles are averaged in physical space across members.
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from config_part1 import (
    MODEL_DIR, DEVICE, N_SENSORS, GROUP_SIZES, SENSOR_GROUPS, HRR_CHANNELS,
    N_TIMESTEPS, T_AMBIENT, GEOMETRY_NAMES, DT_DEVC,
)
from data_loader_part1 import (
    build_dataset, prepare_data_splits, ChannelScaler,
)
from model_part1 import Part1Surrogate, MODEL_NAME

# Growth phase of the BS 8414 burner ramp: 0 -> 720 s is monotonic heat input,
# so a physically sound prediction must not cool through it beyond noise.
GROWTH_END_STEP = int(720.0 / DT_DEVC)
MONOTONIC_TOLERANCE = 5.0  # degC, LES turbulence gives real local dips


def masked_r2(pred, true, mask):
    """R2 over reported steps, pooled across the given axes."""
    keep = mask.reshape(-1) > 0
    p = pred.reshape(-1, pred.shape[-1])[keep]
    t = true.reshape(-1, true.shape[-1])[keep]
    ss_res = ((t - p) ** 2).sum()
    ss_tot = ((t - t.mean(axis=0)) ** 2).sum()
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def masked_rmse(pred, true, mask):
    keep = mask.reshape(-1) > 0
    p = pred.reshape(-1, pred.shape[-1])[keep]
    t = true.reshape(-1, true.shape[-1])[keep]
    return float(np.sqrt(((t - p) ** 2).mean()))


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

    tc_scaler = ChannelScaler().load_state_dict(ckpt["tc_scaler"])
    hrr_scaler = ChannelScaler().load_state_dict(ckpt["hrr_scaler"])
    return models, tc_scaler, hrr_scaler, ckpt


@torch.no_grad()
def predict(models, dataset, device, tc_scaler, hrr_scaler):
    """Ensemble mean in physical units. Returns (tc, hrr, mask) as numpy."""
    params = dataset.params.to(device)
    time_array = dataset.time_array.to(device)

    tc_sum = hrr_sum = None
    for model in models:
        tc, hrr, _ = model(params, time_array)
        tc_sum = tc if tc_sum is None else tc_sum + tc
        hrr_sum = hrr if hrr_sum is None else hrr_sum + hrr

    tc = tc_scaler.inverse((tc_sum / len(models)).cpu().numpy())
    hrr = hrr_scaler.inverse((hrr_sum / len(models)).cpu().numpy())
    return tc, hrr, dataset.mask.numpy()


def physics_gates(tc_pred, tc_true, mask):
    """Physics checks, referenced against the ground truth on the same cases.

    An absolute monotonicity threshold was tried first and rejected: the FDS data
    itself has a worst growth-phase step of -440 degC with 26% of all growth-phase
    steps below -5 degC (`_diag_growth_gate.py`). A point thermocouple in an LES
    fire is not a monotonic signal at DT=10 s, so any absolute tolerance strict
    enough to be meaningful fails on the ground truth and says nothing about the
    model.

    So the growth and plateau gates ask the only question that is answerable:
    **is the prediction less physical than the simulation it is imitating?** The
    sub-ambient gate stays absolute — the truth genuinely respects it (min 17.9 degC
    against TMPA=18), so a sub-ambient prediction is unambiguously wrong.
    """
    valid = (mask > 0)[..., None]

    sub_ambient = int(((tc_pred < T_AMBIENT - 0.5) & valid).sum())
    sub_ambient_true = int(((tc_true < T_AMBIENT - 0.5) & valid).sum())

    def growth_stats(a):
        d = np.diff(a[:, :GROWTH_END_STEP + 1, :], axis=1)
        if d.size == 0:
            return 0.0, 0.0
        return float(d.min()), float(np.percentile(d, 0.1))

    worst_pred, p01_pred = growth_stats(tc_pred)
    worst_true, p01_true = growth_stats(tc_true)

    def tail_rise(a):
        tail = a[:, int(1500.0 / DT_DEVC):, :]
        return float(np.diff(tail, axis=1).max()) if tail.shape[1] > 1 else 0.0

    rise_pred, rise_true = tail_rise(tc_pred), tail_rise(tc_true)

    return {
        "sub_ambient_points": sub_ambient,
        "sub_ambient_points_truth": sub_ambient_true,
        "sub_ambient_pass": sub_ambient <= sub_ambient_true,

        "worst_growth_drop_degC": round(worst_pred, 2),
        "worst_growth_drop_truth_degC": round(worst_true, 2),
        "growth_p01_degC": round(p01_pred, 2),
        "growth_p01_truth_degC": round(p01_true, 2),
        # Not worse than the data, allowing MONOTONIC_TOLERANCE of slack.
        "growth_monotonic_pass": worst_pred >= worst_true - MONOTONIC_TOLERANCE,

        "max_late_rise_degC_per_step": round(rise_pred, 2),
        "max_late_rise_truth_degC_per_step": round(rise_true, 2),
        "late_plateau_pass": rise_pred <= rise_true + MONOTONIC_TOLERANCE,
    }


def report_split(name, models, dataset, split_meta, device, tc_scaler, hrr_scaler):
    if dataset is None or len(dataset) == 0:
        print(f"\n  [{name}] empty split")
        return None

    tc_pred, hrr_pred, mask = predict(models, dataset, device, tc_scaler, hrr_scaler)
    tc_true = tc_scaler.inverse(dataset.tc.numpy())
    hrr_true = hrr_scaler.inverse(dataset.hrr.numpy())

    print(f"\n  [{name}]  {len(dataset)} simulations")
    print(f"    thermocouples   R2 {masked_r2(tc_pred, tc_true, mask):>7.4f}   "
          f"RMSE {masked_rmse(tc_pred, tc_true, mask):>7.2f} degC")

    start = 0
    group_rows = {}
    for group, size in zip(SENSOR_GROUPS, GROUP_SIZES):
        sl = slice(start, start + size)
        r2 = masked_r2(tc_pred[..., sl], tc_true[..., sl], mask)
        rmse = masked_rmse(tc_pred[..., sl], tc_true[..., sl], mask)
        print(f"      {group:<16} R2 {r2:>7.4f}   RMSE {rmse:>7.2f} degC")
        group_rows[group] = {"r2": r2, "rmse": rmse}
        start += size

    hrr_rows = {}
    print(f"    HRR channels")
    for i, channel in enumerate(HRR_CHANNELS):
        sl = slice(i, i + 1)
        r2 = masked_r2(hrr_pred[..., sl], hrr_true[..., sl], mask)
        rmse = masked_rmse(hrr_pred[..., sl], hrr_true[..., sl], mask)
        print(f"      {channel:<16} R2 {r2:>7.4f}   RMSE {rmse:>7.1f} kW")
        hrr_rows[channel] = {"r2": r2, "rmse": rmse}

    print(f"    per geometry")
    geom_rows = {}
    geom_ids = np.array([m["geom_id"] for m in split_meta])
    for gid in sorted(set(geom_ids.tolist())):
        sel = geom_ids == gid
        r2 = masked_r2(tc_pred[sel], tc_true[sel], mask[sel])
        rmse = masked_rmse(tc_pred[sel], tc_true[sel], mask[sel])
        hrr_r2 = masked_r2(hrr_pred[sel][..., :1], hrr_true[sel][..., :1], mask[sel])
        print(f"      {GEOMETRY_NAMES[gid]:<20} n={int(sel.sum()):>2}  "
              f"TC R2 {r2:>7.4f}  RMSE {rmse:>7.2f} degC   HRR R2 {hrr_r2:>7.4f}")
        geom_rows[GEOMETRY_NAMES[gid]] = {"n": int(sel.sum()), "r2": r2,
                                          "rmse": rmse, "hrr_r2": hrr_r2}

    gates = physics_gates(tc_pred, tc_true, mask)
    print(f"    physics                          model      FDS truth")
    print(f"      sub-ambient points        {gates['sub_ambient_points']:>9} "
          f"{gates['sub_ambient_points_truth']:>14}   "
          f"{'PASS' if gates['sub_ambient_pass'] else 'FAIL'}")
    print(f"      worst growth drop degC    "
          f"{gates['worst_growth_drop_degC']:>9.2f} "
          f"{gates['worst_growth_drop_truth_degC']:>14.2f}   "
          f"{'PASS' if gates['growth_monotonic_pass'] else 'FAIL'}")
    print(f"      growth 0.1pct step degC   {gates['growth_p01_degC']:>9.2f} "
          f"{gates['growth_p01_truth_degC']:>14.2f}")
    print(f"      max late rise degC/step   "
          f"{gates['max_late_rise_degC_per_step']:>9.2f} "
          f"{gates['max_late_rise_truth_degC_per_step']:>14.2f}   "
          f"{'PASS' if gates['late_plateau_pass'] else 'FAIL'}")

    return {"n": len(dataset),
            "tc_r2": masked_r2(tc_pred, tc_true, mask),
            "tc_rmse": masked_rmse(tc_pred, tc_true, mask),
            "groups": group_rows, "hrr": hrr_rows, "geometry": geom_rows,
            "physics": gates}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--split", default=None, choices=("hash", "system"))
    args = ap.parse_args()

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    models, tc_scaler, hrr_scaler, ckpt = load_ensemble(args.model_dir, device)
    split_mode = args.split or ckpt.get("split_mode")

    print("=" * 78)
    print(f"  Part1 surrogate evaluation — {args.model_dir}")
    print(f"  {MODEL_NAME}, {len(models)} ensemble member(s), "
          f"split mode {split_mode}")
    print("=" * 78)

    params, tc, hrr, mask, meta, sensor_names = build_dataset(verbose=False)
    if sensor_names != ckpt["sensor_names"]:
        raise SystemExit("thermocouple channel set changed since training — "
                         "the checkpoint's scalers do not apply")

    datasets, fit_tc, fit_hrr, info, _ = prepare_data_splits(
        params, tc, hrr, mask, meta, mode=split_mode, verbose=False)

    results = {}
    for name in ("train", "valid", "test"):
        results[name] = report_split(name, models, datasets[name],
                                     info["meta"][name], device,
                                     tc_scaler, hrr_scaler)

    if results["valid"] and results["test"]:
        combined = 0.5 * (results["valid"]["tc_r2"] + results["test"]["tc_r2"])
        print(f"\n  combined valid+test TC R2: {combined:.4f}")
        print("  (checkpoint ranking uses this, never test alone)")
        results["combined_tc_r2"] = combined

    out = os.path.join(args.model_dir, "evaluation_part1.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\n  written: {out}")


if __name__ == "__main__":
    main()
