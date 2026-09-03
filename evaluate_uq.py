"""FunDiff generative uncertainty quantification — per-pixel predictive std.

FunDiff is a generative model, so uncertainty is intrinsic: draw UQ_N_SAMPLES
independent latent codes from the DiT (conditioned on the same params+slice),
decode each to a full field, and take the per-pixel MEAN (point prediction) and
STD (uncertainty).  No dropout or ensemble needed — this is the natural UQ of a
conditional generative surrogate.

We then score calibration with the Chavare (arXiv:2607.18294) diagnostics
(uq_metrics.py): central-interval coverage vs nominal, sharpness, reliability /
miscalibration area, Gaussian NLL, and the error<->uncertainty correlation.

Per-case mean/std/true fields are saved to outputs/uq/ for the Smokeview
uncertainty overlays (smokeview_output.py --uq).

Run:  python evaluate_uq.py   (requires fae_best.pt + dit_best.pt)
"""
import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from config import (
    MODEL_DIR, OUTPUT_DIR, DEVICE, FIELD_HEIGHT, FIELD_WIDTH, N_TIMESTEPS,
    N_LATENT_TOKENS, EMBED_DIM, SLICE_IDS, N_SAMPLING_STEPS, UQ_N_SAMPLES,
    INFERENCE_SEED,
)
from data_processing.dataset import build_datasets, full_grid_coords
from model.fae import FunctionAutoencoder
from model.dit import DiT, sample_latent
from evaluate import decode_full
import uq_metrics


@torch.no_grad()
def sample_fields(fae, dit, params, slice_id, ambient, device, grid, n_samples, seed0):
    """Draw n_samples decoded fields; return online per-pixel mean & std (physical-normed)."""
    p = params.unsqueeze(0); s = slice_id.unsqueeze(0); amb = ambient.unsqueeze(0)
    n = 0
    mean = torch.zeros(N_TIMESTEPS, FIELD_HEIGHT, FIELD_WIDTH, device=device)
    m2 = torch.zeros_like(mean)
    for k in range(n_samples):
        g = torch.Generator(device=device).manual_seed(seed0 + k)
        z = sample_latent(dit, p, s, N_LATENT_TOKENS, EMBED_DIM,
                          n_steps=N_SAMPLING_STEPS, generator=g)
        field = decode_full(fae, z, amb, device, grid)      # (T,H,W) normalized
        n += 1
        delta = field - mean
        mean += delta / n
        m2 += delta * (field - mean)
    std = torch.sqrt(m2 / max(n - 1, 1))
    return mean.cpu().numpy(), std.cpu().numpy()


def main():
    print("=" * 70)
    print(f"  FunDiff generative UQ  ({UQ_N_SAMPLES} samples/case)")
    print("=" * 70)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    fae_path = os.path.join(MODEL_DIR, "fae_best.pt")
    dit_path = os.path.join(MODEL_DIR, "dit_best.pt")
    if not (os.path.isfile(fae_path) and os.path.isfile(dit_path)):
        print("ERROR: need fae_best.pt + dit_best.pt."); return

    fae_ckpt = torch.load(fae_path, map_location=device, weights_only=False)
    fae = FunctionAutoencoder(config).to(device); fae.load_state_dict(fae_ckpt["fae_state"]); fae.eval()
    dit = DiT(config).to(device)
    dit.load_state_dict(torch.load(dit_path, map_location=device, weights_only=False)["dit_state"]); dit.eval()
    norm_stats = fae_ckpt["norm_stats"]

    _, _, test_ds, _ = build_datasets()
    if len(test_ds) == 0:
        print("  No test functions."); return
    grid = full_grid_coords(device=device)
    uq_dir = os.path.join(OUTPUT_DIR, "uq"); os.makedirs(uq_dir, exist_ok=True)

    per_slice = {}     # slice -> lists of (mean, std, true) physical
    saved = 0
    for i in range(len(test_ds)):
        field, params, slice_id, ambient, chid = test_ds[i]
        sid = int(slice_id); sname = SLICE_IDS[sid]
        st = norm_stats.get(sname, norm_stats["global"]); mu, sd = st["mean"], st["std"]

        mean_n, std_n = sample_fields(
            fae, dit, params.to(device), slice_id.to(device), ambient.to(device),
            device, grid, UQ_N_SAMPLES, INFERENCE_SEED + 1000 * i)
        mean_phys = mean_n * sd + mu
        std_phys = std_n * sd                       # std scales by sd (linear denorm)
        true_phys = field.cpu().numpy() * sd + mu
        per_slice.setdefault(sname, []).append((mean_phys, std_phys, true_phys))

        if saved < 6:
            np.savez_compressed(os.path.join(uq_dir, f"{chid}__{sname}.npz"),
                                mean=mean_phys.astype(np.float32),
                                std=std_phys.astype(np.float32),
                                true=true_phys.astype(np.float32),
                                slice=sname, chid=chid)
            saved += 1

    # Aggregate calibration per slice + global
    print(f"\n  Per-slice generative-UQ calibration (test set):")
    results, allm, alls, allt = {}, [], [], []
    for sname in SLICE_IDS:
        if sname not in per_slice:
            continue
        m = np.stack([a for a, _, _ in per_slice[sname]])
        s = np.stack([b for _, b, _ in per_slice[sname]])
        t = np.stack([c for _, _, c in per_slice[sname]])
        rep = uq_metrics.summarize(t, m, s)
        results[sname] = rep
        allm.append(m.ravel()); alls.append(s.ravel()); allt.append(t.ravel())
        print(uq_metrics.format_report(sname, rep))
    g = uq_metrics.summarize(np.concatenate(allt), np.concatenate(allm), np.concatenate(alls))
    results["GLOBAL"] = g
    print("\n" + uq_metrics.format_report("GLOBAL", g))

    with open(os.path.join(OUTPUT_DIR, "evaluation", "uq_metrics.json"), "w") as f:
        json.dump({"method": "fundiff_generative", "n_samples": UQ_N_SAMPLES,
                   "per_slice": results}, f, indent=2)
    print(f"\n  UQ metrics -> outputs/evaluation/uq_metrics.json")
    print(f"  Saved {saved} mean/std/true field sets -> {uq_dir}")
    print(f"  Overlay with:  python smokeview_output.py --uq")
    print("=" * 70)


if __name__ == "__main__":
    main()
