r"""Post-hoc temperature recalibration for FunDiff generative UQ.

Fits the single Gaussian-NLL-optimal std scale  s = sqrt(mean_valid[(y-mu)^2/sigma^2])
on the VALIDATION split (never test), applies it to the test predictive std, and
re-scores calibration.  Same recipe as the PhysicsNeMo recalibrate.py so the
recalibrated 3-way comparison is consistent.

Run:  venv\Scripts\activate  &&  python recalibrate.py   (needs fae_best.pt+dit_best.pt)
"""
import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from config import (MODEL_DIR, OUTPUT_DIR, DEVICE, N_LATENT_TOKENS, EMBED_DIM,
                    SLICE_IDS, INFERENCE_SEED, UQ_N_SAMPLES)
from data_processing.dataset import build_datasets, full_grid_coords
from model.fae import FunctionAutoencoder
from model.dit import DiT
from evaluate_uq import sample_fields
import uq_metrics


def collect(fae, dit, ds, norm_stats, device, grid, seed_base):
    M, S, Y = [], [], []
    for i in range(len(ds)):
        field, params, slice_id, ambient, chid = ds[i]
        st = norm_stats.get(SLICE_IDS[int(slice_id)], norm_stats["global"])
        mu, sd = st["mean"], st["std"]
        mean_n, std_n = sample_fields(
            fae, dit, params.to(device), slice_id.to(device), ambient.to(device),
            device, grid, UQ_N_SAMPLES, seed_base + 1000 * i)
        M.append((mean_n * sd + mu).ravel())
        S.append((std_n * sd).ravel())
        Y.append((field.cpu().numpy() * sd + mu).ravel())
    return np.concatenate(M), np.concatenate(S), np.concatenate(Y)


def main():
    print("=" * 70)
    print("  FunDiff generative — post-hoc temperature recalibration (valid-fit)")
    print("=" * 70)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    fae_ckpt = torch.load(os.path.join(MODEL_DIR, "fae_best.pt"), map_location=device, weights_only=False)
    fae = FunctionAutoencoder(config).to(device); fae.load_state_dict(fae_ckpt["fae_state"]); fae.eval()
    dit = DiT(config).to(device)
    dit.load_state_dict(torch.load(os.path.join(MODEL_DIR, "dit_best.pt"), map_location=device, weights_only=False)["dit_state"]); dit.eval()
    norm_stats = fae_ckpt["norm_stats"]

    _, valid_ds, test_ds, _ = build_datasets()
    if len(valid_ds) == 0 or len(test_ds) == 0:
        print("  Need valid + test."); return
    grid = full_grid_coords(device=device)

    mv, sv, yv = collect(fae, dit, valid_ds, norm_stats, device, grid, INFERENCE_SEED + 500000)
    sv = np.maximum(sv, 1e-8)
    s = float(np.sqrt(np.mean(((yv - mv) / sv) ** 2)))
    mt, st, yt = collect(fae, dit, test_ds, norm_stats, device, grid, INFERENCE_SEED)
    raw = uq_metrics.summarize(yt, mt, st)
    recal = uq_metrics.summarize(yt, mt, st * s)
    print(f"\n  FunDiff generative: temperature s = {s:.3f} (fit on valid)")
    print(f"    raw   : cov1={raw['coverage_1sigma']:.3f} cov2={raw['coverage_2sigma']:.3f} "
          f"miscal={raw['miscalibration_area']:.3f} NLL={raw['gaussian_nll']:.2f}")
    print(f"    recal : cov1={recal['coverage_1sigma']:.3f} cov2={recal['coverage_2sigma']:.3f} "
          f"miscal={recal['miscalibration_area']:.3f} NLL={recal['gaussian_nll']:.2f}")

    with open(os.path.join(OUTPUT_DIR, "evaluation", "uq_recalibrated.json"), "w") as f:
        json.dump({"fundiff_generative": {"scale": s, "raw": raw, "recalibrated": recal}}, f, indent=2)
    print(f"\n  -> {os.path.join(OUTPUT_DIR, 'evaluation', 'uq_recalibrated.json')}")


if __name__ == "__main__":
    main()
