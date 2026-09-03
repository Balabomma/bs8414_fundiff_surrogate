"""Train the Part1 geometry-variant slice-field surrogate.

Parent recipe: `train.py` (60-sim corpus). Kept: composite MSE + SSIM +
smoothness loss, growth-phase physics penalty, EMA, AMP, gradient accumulation,
multi-member ensembling, early stopping.

Three changes, each forced by the Part1 corpus:

1. **Absent planes are never sampled.** 92 of 185 cases have no ventilated
   cavity, so `Wing_cavity` and `Main_cavity` do not exist for them. Training
   samples are (case, plane) pairs drawn only where `plane_mask` is set.

2. **Case balancing.** A 5-plane case would otherwise contribute 5/3 the
   gradient of a 3-plane case purely because of what FDS wrote to disk — and
   since plane count is perfectly confounded with the `noair` geometry, that
   would systematically under-train exactly the geometry under study. Each
   sample is scaled so every case carries the same total weight. `--no-balance`
   turns this off for a sensitivity check.

3. **The energy constraint is re-based on measured HRR.** The parent correlated
   mean predicted temperature against `params[:, 1]` = normalised burner HRR.
   In Part1 that column is the insulation id and the burner is identical in
   every deck, so the parent's constraint would regress onto a categorical code.
   It now uses each case's actual HRR(t) from `_hrr.csv`, which does vary and is
   precisely the combustion contribution the geometry study is about.

    python -u train_slices_part1.py --model-dir models_part1 > train_slices_part1_run1.log 2> train_slices_part1_run1.err.log
    python -u train_slices_part1.py --split system --model-dir models_part1_system
    python -u train_slices_part1.py --members 5 --model-dir models_part1_ens
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from config_part1 import (
    DEVICE, N_SLICES, SLICE_IDS, SLICE_LOSS_WEIGHTS, N_TIMESTEPS, SPLIT_MODE,
    GEOMETRY_NAMES, T_AMBIENT, PROJECT_DIR,
    SLICE_BATCH_SIZE as BATCH_SIZE, SLICE_ACCUM_STEPS as ACCUM_STEPS,
    SLICE_NUM_EPOCHS as NUM_EPOCHS, SLICE_PATIENCE as PATIENCE,
    SLICE_LEARNING_RATE as LEARNING_RATE, SLICE_WEIGHT_DECAY as WEIGHT_DECAY,
    SLICE_TEMPORAL_WINDOW as TEMPORAL_WINDOW, SLICE_EMA_DECAY as EMA_DECAY,
    SLICE_USE_AMP as USE_AMP,
    LAMBDA_SSIM, LAMBDA_SMOOTH, LAMBDA_GROWTH, LAMBDA_ENERGY,
)
from slice_loader_part1 import (
    load_corpus, load_hrr_curves, prepare_slice_splits, FieldScaler,
)
from slice_losses_part1 import ssim_loss, EMA
from model_part1 import (Part1Surrogate, regularization, count_parameters,
                         MODEL_NAME, LAMBDA_REG)
import physics_part1 as physics

DEFAULT_MODEL_DIR = os.path.join(PROJECT_DIR, "models_part1")


# ──────────────────────────────────────────────────────────────────────
# (case, plane, window) sampling
# ──────────────────────────────────────────────────────────────────────

class PlaneWindowDataset(Dataset):
    """One sample = one existing plane of one case, over a temporal window.

    Windows are strided at build time rather than drawn at random each epoch so
    that validation is deterministic and two runs see identical batches.
    """

    def __init__(self, base, window=TEMPORAL_WINDOW, stride=8, balance_cases=True):
        self.base = base
        self.window = window
        self.samples = []

        plane_w = np.asarray(SLICE_LOSS_WEIGHTS, dtype=np.float32)
        n_cases = len(base)

        for case in range(n_cases):
            present = base.plane_mask[case].numpy() > 0
            if not present.any():
                continue
            # Scale so the present planes of this case average weight 1.0 —
            # relative plane emphasis preserved, case totals equalised.
            case_scale = (float(plane_w[present].mean()) if balance_cases else 1.0)
            n_present = int(present.sum())

            for plane in range(N_SLICES):
                if not present[plane]:
                    continue
                w = float(plane_w[plane]) / case_scale
                if balance_cases:
                    # Equal total weight per case regardless of plane count.
                    w *= N_SLICES / n_present
                valid = base.time_mask[case, plane].numpy() > 0
                for start in range(0, N_TIMESTEPS - window + 1, stride):
                    if valid[start:start + window].all():
                        self.samples.append((case, plane, start, w))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        case, plane, start, w = self.samples[i]
        sl = slice(start, start + self.window)

        field = self.base.fields[case, plane, sl].astype(np.float32)
        field = torch.from_numpy(self.base.scaler.transform(field)).unsqueeze(1)

        return (self.base.params[case], torch.tensor(plane, dtype=torch.long),
                self.base.time_array[sl], field, torch.tensor(w, dtype=torch.float32),
                self.base.hrr_curves[case, sl])


# ──────────────────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────────────────

class Part1SliceLoss(nn.Module):
    """Composite field loss. Weighted per sample; energy term keyed on real HRR."""

    def __init__(self, ambient_normalized=0.0):
        super().__init__()
        self.ambient = ambient_normalized

    def forward(self, pred, target, sample_w, hrr_window):
        B, T = pred.shape[:2]
        w = sample_w.view(B, 1, 1, 1, 1)

        errors = (pred - target) ** 2
        abs_target = target.abs()
        temp_weights = 1.0 + 0.5 * (abs_target / abs_target.max().clamp(min=1.0))
        mse = (errors * temp_weights * w).mean()

        ssim_avg = ssim_loss(pred.reshape(B * T, 1, *pred.shape[3:]),
                             target.reshape(B * T, 1, *target.shape[3:]))

        smooth = (pred[:, 1:] - pred[:, :-1]).pow(2).mean() if T > 1 \
            else torch.zeros((), device=pred.device)

        if T > 2:
            half = T // 2
            growth = F.relu(-(pred[:, 1:half + 1] - pred[:, :half])).pow(2).mean()
        else:
            growth = torch.zeros((), device=pred.device)

        # Energy magnitude: mean predicted field temperature should track the
        # case's measured heat release across the batch. fp32 under autocast.
        energy = torch.zeros((), device=pred.device)
        if B >= 2:
            mean_T = pred.float().mean(dim=[1, 2, 3, 4])
            hrr = hrr_window.float().mean(dim=1)
            mean_c = mean_T - mean_T.mean()
            hrr_c = hrr - hrr.mean()
            denom = hrr_c.pow(2).sum()
            if torch.isfinite(denom) and denom > 1e-4:
                k = (mean_c * hrr_c).sum() / denom
                if torch.isfinite(k):
                    energy = (mean_c - k * hrr_c).pow(2).mean()

        total = (mse + LAMBDA_SSIM * ssim_avg + LAMBDA_SMOOTH * smooth
                 + LAMBDA_GROWTH * growth + LAMBDA_ENERGY * energy)
        return total, {"mse": mse.item(), "ssim": ssim_avg.item(),
                       "smooth": smooth.item(), "growth": growth.item(),
                       "energy": float(energy)}


# ──────────────────────────────────────────────────────────────────────
# Loops
# ──────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, device, optimiser=None, ema=None,
              use_amp=True, accum_steps=1, scaling=None):
    train = optimiser is not None
    model.train() if train else model.eval()
    totals, n_batches = {}, 0

    if train:
        optimiser.zero_grad(set_to_none=True)

    with torch.set_grad_enabled(train):
        for step, batch in enumerate(loader):
            params, plane, times, field, w, hrr = [b.to(device) for b in batch]

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                pred = model(params, plane, times)
                loss, parts = criterion(pred, field, w, hrr)

                # Cavity-gated chimney ordering (physics_part1.py). The field
                # models answer the same counterfactual question as the
                # thermocouple ones — remove the barriers, does the facade get
                # hotter — so the constraint is the identical hinge on the mean
                # predicted field. Defaults to 0.0 = off.
                if scaling is not None and physics.LAMBDA_GEOM:
                    geom_t = physics.geometry_ordering_loss(
                        physics.slice_mean_fn(model, plane, times,
                                              scaling["field_mean"],
                                              scaling["field_scale"]),
                        params,
                        noair_eligible=physics.planes_without_cavity(plane))
                    loss = loss + physics.LAMBDA_GEOM * geom_t
                    parts["geom"] = geom_t.item()

                if train and LAMBDA_REG:
                    loss = loss + LAMBDA_REG * regularization(model)

            if train:
                (loss / accum_steps).backward()
                if (step + 1) % accum_steps == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimiser.step()
                    optimiser.zero_grad(set_to_none=True)
                    if ema is not None:
                        ema.update(model)

            parts["loss"] = loss.item()
            for k, v in parts.items():
                totals[k] = totals.get(k, 0.0) + v
            n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


def train_member(train_ds, valid_ds, device, seed, epochs, use_amp,
                 scaling=None, log_every=5):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = Part1Surrogate().to(device)
    criterion = Part1SliceLoss().to(device)
    ema = EMA(model, decay=EMA_DECAY)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE) if valid_ds else None

    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=10)

    best = {"loss": float("inf"), "epoch": -1, "state": None}
    since_improved, history = 0, []
    eval_model = Part1Surrogate().to(device)

    for epoch in range(epochs):
        tr = run_epoch(model, train_loader, criterion, device, optimiser, ema,
                       use_amp, ACCUM_STEPS, scaling=scaling)

        if valid_loader:
            eval_model.load_state_dict(ema.apply_to(model))
            va = run_epoch(eval_model, valid_loader, criterion, device,
                           use_amp=use_amp, scaling=scaling)
        else:
            va = tr

        scheduler.step(va["loss"])
        history.append({"epoch": epoch, "train": tr, "valid": va,
                        "lr": optimiser.param_groups[0]["lr"]})

        if va["loss"] < best["loss"] - 1e-6:
            best = {"loss": va["loss"], "epoch": epoch,
                    "state": {k: v.detach().cpu().clone()
                              for k, v in ema.apply_to(model).items()}}
            since_improved = 0
        else:
            since_improved += 1

        if epoch % log_every == 0 or since_improved == 0:
            print(f"    epoch {epoch:>4}  train {tr['loss']:.5f} "
                  f"(mse {tr['mse']:.5f} ssim {tr['ssim']:.4f} "
                  f"growth {tr['growth']:.5f})  valid {va['loss']:.5f}"
                  f"  lr {optimiser.param_groups[0]['lr']:.2e}"
                  f"{'  *' if since_improved == 0 else ''}", flush=True)

        if since_improved >= PATIENCE:
            print(f"    early stop at epoch {epoch} "
                  f"(best {best['loss']:.5f} @ {best['epoch']})")
            break

    model.load_state_dict(best["state"])
    return model, best, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--split", default=SPLIT_MODE, choices=("hash", "system"))
    ap.add_argument("--members", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    ap.add_argument("--window", type=int, default=TEMPORAL_WINDOW)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--no-balance", action="store_true",
                    help="do not equalise case weight across plane counts")
    ap.add_argument("--max-sims", type=int, default=None, help="smoke testing")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if (os.path.isdir(args.model_dir)
            and any(f.endswith(".pt") for f in os.listdir(args.model_dir))
            and not args.force):
        raise SystemExit(f"{args.model_dir} already holds checkpoints. "
                         f"Rename it or pass --force.")
    os.makedirs(args.model_dir, exist_ok=True)

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("CUDA not available — refusing to train on CPU.")
    print(f"  device: {torch.cuda.get_device_name(0)}")

    print("\n  loading slice corpus ...")
    params, fields, time_mask, plane_mask, meta = load_corpus(
        max_sims=args.max_sims)
    hrr_curves = load_hrr_curves(meta)
    datasets, scaler, info, _ = prepare_slice_splits(
        params, fields, time_mask, plane_mask, meta, mode=args.split,
        hrr_curves=hrr_curves)

    balance = not args.no_balance
    windowed = {}
    for name in ("train", "valid", "test"):
        windowed[name] = (PlaneWindowDataset(datasets[name], args.window,
                                             args.stride, balance)
                          if datasets[name] is not None else None)
        if windowed[name] is not None:
            print(f"    {name:<6} {len(datasets[name]):>3} cases -> "
                  f"{len(windowed[name]):>5} (plane, window) samples")

    print(f"\n  model: {MODEL_NAME}  ({count_parameters(Part1Surrogate()):,} params)")
    print(f"  case balancing: {'on' if balance else 'OFF'}")
    print(f"  loss: mse + {LAMBDA_SSIM}*ssim + {LAMBDA_SMOOTH}*smooth "
          f"+ {LAMBDA_GROWTH}*growth + {LAMBDA_ENERGY}*energy(measured HRR)")
    print(f"  {physics.describe()}")
    # Field scaler is a single global (mean, sd) in degC, so the ordering
    # margins can be applied directly once the prediction is un-scaled.
    scaling = {"field_mean": float(scaler.mean), "field_scale": float(scaler.scale)}

    members = []
    for i in range(args.members):
        seed = args.seed + i
        print(f"\n  === member {i + 1}/{args.members} (seed {seed}) ===")
        t0 = time.time()
        model, best, history = train_member(
            windowed["train"], windowed["valid"], device, seed, args.epochs,
            USE_AMP, scaling=scaling)

        path = os.path.join(args.model_dir, f"part1_slice_member{i}.pt")
        torch.save({
            "state_dict": model.state_dict(),
            "model_name": MODEL_NAME,
            "seed": seed,
            "best_valid_loss": best["loss"],
            "best_epoch": best["epoch"],
            "field_scaler": scaler.state_dict(),
            "lambda_geom": physics.LAMBDA_GEOM,
            "slice_ids": SLICE_IDS,
            "split_mode": args.split,
            "window": args.window,
            "balance_cases": balance,
        }, path)
        members.append({"path": path, "seed": seed, "valid_loss": best["loss"],
                        "best_epoch": best["epoch"],
                        "minutes": round((time.time() - t0) / 60, 1)})
        print(f"    saved {path}  ({members[-1]['minutes']} min)")

        with open(os.path.join(args.model_dir, f"history_member{i}.json"), "w") as f:
            json.dump(history, f, indent=1)

    with open(os.path.join(args.model_dir, "run_summary.json"), "w") as f:
        json.dump({
            "model_name": MODEL_NAME, "split_mode": args.split,
            "balance_cases": balance, "window": args.window,
            "n_train_cases": len(info["meta"]["train"]),
            "n_valid_cases": len(info["meta"]["valid"]),
            "n_test_cases": len(info["meta"]["test"]),
            "test_chids": [m["chid"] for m in info["meta"]["test"]],
            "members": members,
        }, f, indent=1)

    print(f"\n  done. {len(members)} member(s) in {args.model_dir}")
    print(f"  next: python evaluate_slices_part1.py --model-dir {args.model_dir}")


if __name__ == "__main__":
    main()
