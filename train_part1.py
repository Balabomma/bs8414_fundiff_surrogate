"""Train the Part1 geometry-variant KAN surrogate.

Parent recipe: `train.py` (60-sim corpus). Kept: AdamW, plateau LR schedule,
gradient clipping, early stopping on validation loss, the spline regulariser.
Changed: two output heads, mask-aware losses, and scalers fitted on the training
split only.

    python -u train_part1.py --model-dir models_part1 > train_part1_run1.log 2> train_part1_run1.err.log
    python -u train_part1.py --split system --model-dir models_part1_system
    python -u train_part1.py --members 5 --model-dir models_part1_ens

Checkpoints are written to a NAMED directory that must not already hold a run —
three retrains were lost to silent overwrites on this corpus's predecessor, so
the script refuses rather than clobbers (`--force` to override deliberately).
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config_part1 import (
    MODEL_DIR, DEVICE, BATCH_SIZE, NUM_EPOCHS, PATIENCE, LEARNING_RATE,
    WEIGHT_DECAY, LAMBDA_HRR, N_SENSORS, N_HRR_CHANNELS, GROUP_SIZES,
    SENSOR_GROUPS, HRR_CHANNELS, HRR_NONNEGATIVE, SPLIT_MODE,
)
from data_loader_part1 import build_dataset, prepare_data_splits
from model_part1 import (Part1Surrogate, regularization, count_parameters,
                         MODEL_NAME, LAMBDA_REG)
import physics_part1 as physics


def masked_mse(pred, target, mask):
    """MSE over reported timesteps only.

    mask is (B, T); padded tails of a short run must not be scored, or the model
    is rewarded for reproducing a held last value that the simulation never
    actually wrote.
    """
    m = mask.unsqueeze(-1)
    err = ((pred - target) ** 2) * m
    denom = m.sum() * pred.shape[-1]
    return err.sum() / denom.clamp(min=1.0)


def run_epoch(model, loader, optimiser, device, train, scaling=None):
    model.train() if train else model.eval()
    totals = {"loss": 0.0, "tc": 0.0, "hrr": 0.0, "closure": 0.0, "geom": 0.0,
              "n": 0}

    with torch.set_grad_enabled(train):
        for params, tc, hrr, mask, time_array in loader:
            params = params.to(device)
            tc, hrr, mask = tc.to(device), hrr.to(device), mask.to(device)
            time_array = time_array[0].to(device)

            tc_pred, hrr_pred, _ = model(params, time_array)
            tc_loss = masked_mse(tc_pred, tc, mask)
            hrr_loss = masked_mse(hrr_pred, hrr, mask)
            loss = tc_loss + LAMBDA_HRR * hrr_loss

            # Physics constraints, measured in the FDS truth before being
            # imposed (see physics_part1.py). Both default to 0.0 = off, so a
            # run that does not opt in is arithmetically identical to the
            # recipe every existing Part1 checkpoint was trained under.
            closure = geom_order = 0.0
            if scaling is not None and (physics.LAMBDA_CLOSURE
                                        or physics.LAMBDA_GEOM):
                if physics.LAMBDA_CLOSURE:
                    closure_t = physics.energy_closure_loss(
                        hrr_pred, hrr, mask,
                        scaling["hrr_mean"], scaling["hrr_scale"])
                    loss = loss + physics.LAMBDA_CLOSURE * closure_t
                    closure = closure_t.item()
                if physics.LAMBDA_GEOM:
                    geom_t = physics.geometry_ordering_loss(
                        physics.tc_mean_fn(model, time_array,
                                           scaling["tc_mean"],
                                           scaling["tc_scale"]),
                        params)
                    loss = loss + physics.LAMBDA_GEOM * geom_t
                    geom_order = geom_t.item()

            if train:
                # The regulariser is architecture-specific and may be absent
                # (LAMBDA_REG = 0.0 for the MLP, V3 and Samba variants). Guard
                # ONLY the extra term — never the optimiser step, or a model
                # without a regulariser silently never trains.
                if LAMBDA_REG:
                    loss = loss + LAMBDA_REG * regularization(model)
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()

            n = params.shape[0]
            totals["loss"] += loss.item() * n
            totals["tc"] += tc_loss.item() * n
            totals["hrr"] += hrr_loss.item() * n
            totals["closure"] += closure * n
            totals["geom"] += geom_order * n
            totals["n"] += n

    n = max(totals["n"], 1)
    return {k: totals[k] / n
            for k in ("loss", "tc", "hrr", "closure", "geom")}


def train_member(datasets, tc_scaler, hrr_scaler, device, seed, epochs, log_every=10):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = Part1Surrogate().to(device)
    model.set_output_scaling(
        tc_scaler, hrr_scaler,
        hrr_nonnegative_idx=[HRR_CHANNELS.index(c) for c in HRR_NONNEGATIVE])

    # Physical-space statistics for the physics constraints. The heads are
    # trained standardised, but energy closure and the chimney ordering are
    # statements about kW and degC, so they are formed after un-scaling.
    def _t(a):
        return torch.as_tensor(a, dtype=torch.float32, device=device)
    scaling = {"tc_mean": _t(tc_scaler.mean), "tc_scale": _t(tc_scaler.scale),
               "hrr_mean": _t(hrr_scaler.mean), "hrr_scale": _t(hrr_scaler.scale)}

    train_loader = DataLoader(datasets["train"], batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(datasets["valid"], batch_size=BATCH_SIZE) \
        if datasets["valid"] is not None else None

    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=15)

    best = {"loss": float("inf"), "epoch": -1, "state": None}
    since_improved = 0
    history = []

    # Sentinel: snapshot the weights so the first epoch can prove it moved them.
    # A silently-disabled optimiser step reads exactly like instant convergence —
    # flat loss, "best @ epoch 0" — and cost a full 3-member run before it was
    # caught. Cheap to check, so it is checked every run.
    before = torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()

    for epoch in range(epochs):
        tr = run_epoch(model, train_loader, optimiser, device, train=True,
                       scaling=scaling)

        if epoch == 0:
            after = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
            delta = (after - before).norm().item()
            if delta < 1e-8:
                raise RuntimeError(
                    f"no weight change after one epoch (|dW| = {delta:.3e}) — "
                    f"the optimiser is not stepping. Training would report flat "
                    f"loss and 'best @ epoch 0' as if it had converged.")
            print(f"    [sentinel] weights moved after epoch 0: "
                  f"|dW| = {delta:.4e}")
        va = run_epoch(model, valid_loader, None, device, train=False,
                       scaling=scaling) \
            if valid_loader else tr
        scheduler.step(va["loss"])
        history.append({"epoch": epoch, "train": tr, "valid": va,
                        "lr": optimiser.param_groups[0]["lr"]})

        if va["loss"] < best["loss"] - 1e-5:
            best = {"loss": va["loss"], "epoch": epoch,
                    "state": {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}}
            since_improved = 0
        else:
            since_improved += 1

        if epoch % log_every == 0 or since_improved == 0:
            print(f"    epoch {epoch:>4}  train {tr['loss']:.5f} "
                  f"(tc {tr['tc']:.5f} hrr {tr['hrr']:.5f})  "
                  f"valid {va['loss']:.5f} (tc {va['tc']:.5f} hrr {va['hrr']:.5f})"
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
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--split", default=SPLIT_MODE, choices=("hash", "system"))
    ap.add_argument("--members", type=int, default=1,
                    help="ensemble members; each gets its own seed and checkpoint")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    ap.add_argument("--force", action="store_true",
                    help="allow writing into a directory that already has a run")
    args = ap.parse_args()

    if (os.path.isdir(args.model_dir)
            and any(f.endswith(".pt") for f in os.listdir(args.model_dir))
            and not args.force):
        raise SystemExit(
            f"{args.model_dir} already holds checkpoints. Rename it or pass "
            f"--force. (Overwriting has already cost this project three "
            f"retrains — see VARIANCE_RECORD.md.)")
    os.makedirs(args.model_dir, exist_ok=True)

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("CUDA not available — refusing to train on CPU. "
                         "Check the venv's torch build and nvidia-smi.")
    print(f"  device: {torch.cuda.get_device_name(0)}")

    print("\n  building dataset ...")
    params, tc, hrr, mask, meta, sensor_names = build_dataset()
    datasets, tc_scaler, hrr_scaler, info, time_array = prepare_data_splits(
        params, tc, hrr, mask, meta, mode=args.split)

    print(f"\n  target: {N_SENSORS} thermocouples "
          f"({' + '.join(f'{k} x{v}' for k, v in SENSOR_GROUPS.items())})")
    print(f"          {N_HRR_CHANNELS} HRR channels ({', '.join(HRR_CHANNELS)})")
    print(f"  model:  {MODEL_NAME}")
    print(f"  loss:   tc + {LAMBDA_HRR} * hrr"
          + (f" + {LAMBDA_REG} * reg" if LAMBDA_REG else "  (no weight penalty)"))
    print(f"  {physics.describe()}")

    members = []
    for i in range(args.members):
        seed = args.seed + i
        print(f"\n  === member {i + 1}/{args.members} (seed {seed}) ===")
        t0 = time.time()
        model, best, history = train_member(datasets, tc_scaler, hrr_scaler,
                                            device, seed, args.epochs)
        path = os.path.join(args.model_dir, f"part1_member{i}.pt")
        torch.save({
            "state_dict": model.state_dict(),
            "model_name": MODEL_NAME,
            "seed": seed,
            "best_valid_loss": best["loss"],
            "best_epoch": best["epoch"],
            "tc_scaler": tc_scaler.state_dict(),
            "hrr_scaler": hrr_scaler.state_dict(),
            "sensor_names": sensor_names,
            "hrr_channels": HRR_CHANNELS,
            "split_mode": args.split,
            "n_params": count_parameters(model),
        }, path)
        members.append({"path": path, "seed": seed, "valid_loss": best["loss"],
                        "best_epoch": best["epoch"],
                        "minutes": round((time.time() - t0) / 60, 1)})
        print(f"    saved {path}  ({members[-1]['minutes']} min)")

        with open(os.path.join(args.model_dir, f"history_member{i}.json"), "w") as f:
            json.dump(history, f, indent=1)

    summary = {
        "model_name": MODEL_NAME,
        "split_mode": args.split,
        "n_train": len(info["meta"]["train"]),
        "n_valid": len(info["meta"]["valid"]),
        "n_test": len(info["meta"]["test"]),
        "sensor_names": sensor_names,
        "hrr_channels": HRR_CHANNELS,
        "lambda_hrr": LAMBDA_HRR,
        "lambda_closure": physics.LAMBDA_CLOSURE,
        "lambda_geom": physics.LAMBDA_GEOM,
        "members": members,
        "test_chids": [m["chid"] for m in info["meta"]["test"]],
        "valid_chids": [m["chid"] for m in info["meta"]["valid"]],
    }
    with open(os.path.join(args.model_dir, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    print(f"\n  done. {len(members)} member(s) in {args.model_dir}")
    print(f"  next: python evaluate_part1.py --model-dir {args.model_dir}")


if __name__ == "__main__":
    main()
