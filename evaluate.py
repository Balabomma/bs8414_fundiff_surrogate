"""Evaluation for the FunDiff surrogate — deterministic conditional-mean readout.

Because the FDS field is (near-)deterministic given the deck, the generative
model is read out deterministically: for each test (sim, slice) we condition the
DiT on (params, slice), sample N_COND_SAMPLES latents from fixed-seed Gaussian
noise, average them (the conditional mean), and decode the full 181x128x64 grid
via the FAE decoder.  All metrics are in physical degrees C on the fixed
42/9/9 split — directly comparable to the other slice surrogates.

Two readouts are reported to separate autoencoder quality from generative quality
(the point of the staged design):
    FAE(recon)  : encode ground truth -> decode      (upper bound / Stage-1 only)
    DiT(gen)    : sample latent -> decode             (full FunDiff)
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from config import (
    MODEL_DIR, OUTPUT_DIR, DEVICE, FIELD_HEIGHT, FIELD_WIDTH, N_TIMESTEPS,
    N_LATENT_TOKENS, EMBED_DIM, SLICE_IDS, SLICE_ID_MAP,
    N_SAMPLING_STEPS, N_COND_SAMPLES, INFERENCE_SEED, DECODE_CHUNK,
)
# Part1 corpus: data layer swapped, coordinate helper is corpus-independent.
from dataset_part1 import build_datasets
from data_processing.dataset import full_grid_coords
from model.fae import FunctionAutoencoder, count_parameters
from model.dit import sample_latent
# Uniform Part1 interface: Part1DiT here, Part1KANDiT in the KAN variant.
from model_part1 import Part1Surrogate, MODEL_NAME
from train_fae import ssim_loss


# MAPE is meaningless where the denominator approaches ambient, and most of a
# facade slice sits at ambient for most of the run. It is therefore reported
# over cells at or above this temperature only, matching the floor the sensor
# evaluation uses so the two granularities are comparable.
MAPE_FLOOR_C = 100.0


def metrics(pred, actual):
    pred = pred.astype(np.float64); actual = actual.astype(np.float64)
    ss_res = np.sum((pred - actual) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    err = pred - actual
    sel = actual >= MAPE_FLOOR_C
    out = {
        "r2": float(1 - ss_res / max(ss_tot, 1e-10)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "mbe": float(np.mean(err)),
        "mape_pct": float(np.mean(np.abs(err[sel] / actual[sel])) * 100.0) if sel.any() else float("nan"),
        "n_mape": int(sel.sum()),
        "mape_floor_c": MAPE_FLOOR_C,
        "p95_abs_err": float(np.percentile(np.abs(err), 95)),
    }
    return out


@torch.no_grad()
def decode_full(fae, latent, ambient, device, grid):
    preds = []
    for c0 in range(0, grid.shape[0], DECODE_CHUNK):
        cc = grid[c0:c0 + DECODE_CHUNK].unsqueeze(0)
        preds.append(fae.decode(latent, cc, ambient).squeeze(0))
    return torch.cat(preds).reshape(N_TIMESTEPS, FIELD_HEIGHT, FIELD_WIDTH)


def frame_ssim(pred_norm, true_norm):
    """Mean SSIM over frames (on normalized fields, matching training)."""
    p = torch.from_numpy(pred_norm).unsqueeze(1)
    t = torch.from_numpy(true_norm).unsqueeze(1)
    vals = []
    for i in range(0, p.shape[0], 32):
        vals.append(1.0 - ssim_loss(p[i:i + 32], t[i:i + 32]).item())
    return float(np.mean(vals))


def main():
    print("=" * 70)
    print("  BS8414 FunDiff surrogate — Evaluation (conditional-mean readout)")
    print("=" * 70)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    fae_path = os.path.join(MODEL_DIR, "fae_best.pt")
    dit_path = os.path.join(MODEL_DIR, "dit_best.pt")
    if not os.path.isfile(fae_path):
        print("ERROR: fae_best.pt not found — run train_fae.py."); return
    fae_ckpt = torch.load(fae_path, map_location=device, weights_only=False)
    fae = FunctionAutoencoder(config).to(device)
    fae.load_state_dict(fae_ckpt["fae_state"]); fae.eval()
    norm_stats = fae_ckpt["norm_stats"]

    have_dit = os.path.isfile(dit_path)
    if have_dit:
        dit = Part1Surrogate(config).to(device)
        dit.load_state_dict(torch.load(dit_path, map_location=device,
                                       weights_only=False)["dit_state"])
        dit.eval()
        print(f"  Loaded FAE ({count_parameters(fae):,} p) + DiT ({count_parameters(dit):,} p)")
    else:
        print(f"  Loaded FAE ({count_parameters(fae):,} p).  No DiT yet — FAE-only eval.")

    _, _, test_ds, _ = build_datasets()
    if len(test_ds) == 0:
        print("  No test functions."); return

    grid = full_grid_coords(device=device)
    fields_dir = os.path.join(OUTPUT_DIR, "fields"); os.makedirs(fields_dir, exist_ok=True)

    recon, gen, saved = {}, {}, []
    for i in range(len(test_ds)):
        field, params, slice_id, ambient, chid = test_ds[i]
        sid = int(slice_id); sname = SLICE_IDS[sid]
        stats = norm_stats.get(sname, norm_stats["global"])
        mean, std = stats["mean"], stats["std"]

        field = field.to(device); amb = ambient.to(device).unsqueeze(0)
        p = params.to(device).unsqueeze(0); s = slice_id.to(device).unsqueeze(0)
        true_norm = field.cpu().numpy()
        true_phys = true_norm * std + mean

        # FAE reconstruction readout
        z_recon = fae.encode(field.unsqueeze(0))
        pr_norm = decode_full(fae, z_recon, amb, device, grid).cpu().numpy()
        recon.setdefault(sname, []).append((pr_norm * std + mean, true_phys))

        # DiT generative conditional-mean readout
        if have_dit:
            g = torch.Generator(device=device).manual_seed(INFERENCE_SEED + i)
            zs = [sample_latent(dit, p, s, N_LATENT_TOKENS, EMBED_DIM,
                                n_steps=N_SAMPLING_STEPS, generator=g)
                  for _ in range(N_COND_SAMPLES)]
            z_mean = torch.stack(zs).mean(0)
            pg_norm = decode_full(fae, z_mean, amb, device, grid).cpu().numpy()
            pg_phys = pg_norm * std + mean
            gen.setdefault(sname, []).append((pg_phys, true_phys))
        else:
            pg_phys = pr_norm * std + mean  # fallback for field export

        # save fields for Smokeview rendering (a handful)
        if len(saved) < 6:
            fp = os.path.join(fields_dir, f"{chid}__{sname}.npz")
            np.savez_compressed(fp, pred=pg_phys.astype(np.float32),
                                true=true_phys.astype(np.float32),
                                slice=sname, chid=chid)
            saved.append(fp)

    def summarize(store, tag):
        print(f"\n  {tag}")
        print(f"  {'Slice':<18s}{'R2':>9s}{'RMSE(C)':>10s}{'MAE(C)':>9s}{'SSIM':>8s}")
        print("  " + "-" * 54)
        allp, allt, per = [], [], {}
        for sname in SLICE_IDS:
            if sname not in store:
                continue
            preds = np.stack([a for a, _ in store[sname]])
            trues = np.stack([b for _, b in store[sname]])
            m = metrics(preds, trues)
            pn = (preds - norm_stats[sname]["mean"]) / norm_stats[sname]["std"]
            tn = (trues - norm_stats[sname]["mean"]) / norm_stats[sname]["std"]
            ss = frame_ssim(pn.reshape(-1, FIELD_HEIGHT, FIELD_WIDTH),
                            tn.reshape(-1, FIELD_HEIGHT, FIELD_WIDTH))
            m["ssim"] = ss; per[sname] = m
            print(f"  {sname:<18s}{m['r2']:9.4f}{m['rmse']:10.2f}{m['mae']:9.2f}{ss:8.3f}")
            allp.append(preds.ravel()); allt.append(trues.ravel())
        g = metrics(np.concatenate(allp), np.concatenate(allt))
        print(f"  {'GLOBAL':<18s}{g['r2']:9.4f}{g['rmse']:10.2f}{g['mae']:9.2f}")
        return {"global": g, "per_slice": per}

    results = {"fae_recon": summarize(recon, "FAE(recon) — encode GT -> decode  [Stage-1 upper bound]")}
    if have_dit:
        results["dit_gen"] = summarize(gen, "DiT(gen) — sample latent -> decode  [full FunDiff]")

    out = os.path.join(OUTPUT_DIR, "evaluation"); os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    # A copy in the model directory too: the shared path above is overwritten by
    # every evaluation, so without this a replicate's metrics are lost as soon as
    # the next one runs.
    try:
        with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        print("  could not write per-model metrics:", e)
    print(f"\n  Metrics -> {os.path.join(out, 'metrics.json')}")
    print(f"  Saved {len(saved)} pred/true field pairs -> {fields_dir}")
    print(f"  Render Smokeview-style output with:  python smokeview_output.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
