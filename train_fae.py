r"""Stage 1 — train the Function Autoencoder (FAE).

The FAE is trained to reconstruct the spatiotemporal temperature field at
randomly sampled continuous (t, y, z) query points (paper's L_FAE), plus:
  - an SSIM term on a few fully-decoded frames (structural fidelity),
  - the growth-phase monotonicity and energy/HRR physics priors,
  - a hard ambient floor baked into the decoder.

The trained FAE is BOTH the Stage-2 latent provider AND a usable standalone
deterministic surrogate (encode ground truth -> decode) — reported here via a
full-grid reconstruction R2 on the held-out split.

Run:  venv\Scripts\activate  &&  python train_fae.py
Logs: tee to train_fae_run1.log / .err.log (see the bs8414-surrogates skill).
"""
import os
import sys
import copy
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from config import (
    MODEL_DIR, OUTPUT_DIR, DEVICE, FIELD_HEIGHT, FIELD_WIDTH, N_TIMESTEPS,
    FAE_LR, FAE_WARMUP_STEPS, FAE_DECAY_EVERY, FAE_DECAY_RATE, FAE_WEIGHT_DECAY,
    FAE_BATCH_SIZE, FAE_N_QUERY, FAE_EPOCHS, FAE_PATIENCE, FAE_EMA_DECAY,
    FAE_ENC_SUBSAMPLE_T, LAMBDA_SSIM, LAMBDA_GROWTH, LAMBDA_ENERGY,
    SSIM_EVAL_FRAMES, GROWTH_PHASE_FRAC, DECODE_CHUNK,
)
# Part1 corpus: the data layer is swapped, the coordinate helpers are not -
# they are geometry utilities and are corpus-independent.
from dataset_part1 import build_datasets, collate
from data_processing.dataset import sample_query_points, full_grid_coords
from model.fae import FunctionAutoencoder, count_parameters
from model.physics import growth_monotonicity_loss, energy_hrr_loss


# ── SSIM ─────────────────────────────────────────────────────────────

def _gauss_kernel(size, sigma, device):
    c = torch.arange(size, device=device).float() - size // 2
    g = torch.exp(-(c ** 2) / (2 * sigma ** 2)); g /= g.sum()
    k = (g[:, None] * g[None, :])
    return k[None, None]


def ssim_loss(pred, target, size=7, sigma=1.5):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    k = _gauss_kernel(size, sigma, pred.device)
    pad = size // 2
    mp = F.conv2d(pred, k, padding=pad); mt = F.conv2d(target, k, padding=pad)
    spp = F.conv2d(pred * pred, k, padding=pad) - mp * mp
    stt = F.conv2d(target * target, k, padding=pad) - mt * mt
    spt = F.conv2d(pred * target, k, padding=pad) - mp * mt
    ssim = ((2 * mp * mt + C1) * (2 * spt + C2)) / ((mp * mp + mt * mt + C1) * (spp + stt + C2))
    return 1.0 - ssim.mean()


# ── EMA ──────────────────────────────────────────────────────────────

class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def state_dict(self, model):
        sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        for k in self.shadow:
            sd[k] = self.shadow[k].clone()
        return sd


# ── Full-frame coords for the SSIM term ──────────────────────────────

def _frame_coords(device):
    yy = torch.linspace(0, 1, FIELD_HEIGHT, device=device)
    zz = torch.linspace(0, 1, FIELD_WIDTH, device=device)
    gy, gz = torch.meshgrid(yy, zz, indexing="ij")
    return gy.reshape(-1), gz.reshape(-1)          # (H*W,), (H*W,)


def lr_lambda(step):
    if step < FAE_WARMUP_STEPS:
        return (step + 1) / FAE_WARMUP_STEPS
    return FAE_DECAY_RATE ** ((step - FAE_WARMUP_STEPS) / FAE_DECAY_EVERY)


def fae_step(model, fields, params, slice_ids, ambient, device, frame_gy, frame_gz):
    B = fields.shape[0]
    # resolution-invariance augmentation: subsample encoder input in time
    factor = int(np.random.choice(FAE_ENC_SUBSAMPLE_T))
    enc_field = fields[:, ::factor] if factor > 1 else fields

    latent = model.encode(enc_field)

    # 1. point reconstruction MSE
    coords, target = sample_query_points(fields, FAE_N_QUERY, device)
    pred = model.decode(latent, coords, ambient)
    mse = F.mse_loss(pred, target)

    # 2. SSIM on a few fully-decoded frames
    ssim = fields.new_zeros(())
    if LAMBDA_SSIM > 0:
        HW = FIELD_HEIGHT * FIELD_WIDTH
        t_ids = torch.randint(0, N_TIMESTEPS, (SSIM_EVAL_FRAMES,))
        for ti in t_ids.tolist():
            tval = ti / max(N_TIMESTEPS - 1, 1)
            fc = torch.stack([
                torch.full_like(frame_gy, tval), frame_gy, frame_gz], dim=-1)  # (HW,3)
            fc = fc.unsqueeze(0).expand(B, -1, -1)
            fp = model.decode(latent, fc, ambient).reshape(B, 1, FIELD_HEIGHT, FIELD_WIDTH)
            ft = fields[:, ti].unsqueeze(1)
            ssim = ssim + ssim_loss(fp, ft)
        ssim = ssim / SSIM_EVAL_FRAMES

    # 3. physics priors
    growth = growth_monotonicity_loss(
        model.decode, latent, coords, ambient, GROWTH_PHASE_FRAC) if LAMBDA_GROWTH > 0 \
        else fields.new_zeros(())
    energy = energy_hrr_loss(pred, params[:, 1]) if LAMBDA_ENERGY > 0 \
        else fields.new_zeros(())

    total = mse + LAMBDA_SSIM * ssim + LAMBDA_GROWTH * growth + LAMBDA_ENERGY * energy
    return total, {"mse": mse.item(), "ssim": float(ssim.detach()),
                   "growth": float(growth.detach()), "energy": float(energy.detach())}


@torch.no_grad()
def validate(model, loader, device, frame_gy, frame_gz):
    model.eval()
    tot, n = 0.0, 0
    for fields, params, slice_ids, ambient, _ in loader:
        fields = fields.to(device); params = params.to(device)
        slice_ids = slice_ids.to(device); ambient = ambient.to(device)
        latent = model.encode(fields)
        coords, target = sample_query_points(fields, FAE_N_QUERY, device)
        pred = model.decode(latent, coords, ambient)
        tot += F.mse_loss(pred, target).item(); n += 1
    return tot / max(n, 1)


@torch.no_grad()
def full_grid_r2(model, dataset, device):
    """Reconstruction R2 over the full native grid (standalone-FAE metric)."""
    if len(dataset) == 0:
        return None
    grid = full_grid_coords(device=device)             # (T*H*W, 3)
    ss_res, ss_tot, all_mean, cnt = 0.0, 0.0, 0.0, 0
    # first pass mean
    fields_list = []
    for i in range(len(dataset)):
        f, p, s, amb, _ = dataset[i]
        fields_list.append((f.to(device), amb.to(device)))
        all_mean += f.sum().item(); cnt += f.numel()
    gmean = all_mean / cnt
    for f, amb in fields_list:
        latent = model.encode(f.unsqueeze(0))
        preds = []
        for c0 in range(0, grid.shape[0], DECODE_CHUNK):
            cc = grid[c0:c0 + DECODE_CHUNK].unsqueeze(0)
            preds.append(model.decode(latent, cc, amb.unsqueeze(0)).squeeze(0))
        pred = torch.cat(preds).reshape(N_TIMESTEPS, FIELD_HEIGHT, FIELD_WIDTH)
        ss_res += ((pred - f) ** 2).sum().item()
        ss_tot += ((f - gmean) ** 2).sum().item()
    return 1 - ss_res / max(ss_tot, 1e-10)


def main():
    print("=" * 70)
    print("  BS8414 FunDiff surrogate — Stage 1: Function Autoencoder")
    print("=" * 70)
    if DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    device = torch.device(DEVICE)
    if device.type == "cuda":
        print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    torch.backends.cudnn.benchmark = True
    os.makedirs(MODEL_DIR, exist_ok=True); os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_ds, valid_ds, test_ds, norm_stats = build_datasets()
    if len(train_ds) == 0:
        print("ERROR: no training functions found."); return
    train_loader = DataLoader(train_ds, batch_size=FAE_BATCH_SIZE, shuffle=True,
                              collate_fn=collate, num_workers=0, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=FAE_BATCH_SIZE, shuffle=False,
                              collate_fn=collate) if len(valid_ds) else None

    # Seed is env-overridable so independent replicates can be trained without
    # editing the tracked default of 42.
    _seed = int(__import__('os').environ.get('FUNDIFF_SEED', 42))
    torch.manual_seed(_seed); np.random.seed(_seed)
    print(f'  seed: {_seed}')
    model = FunctionAutoencoder(config).to(device)
    print(f"  FAE params: {count_parameters(model):,}")
    print(f"  Train {len(train_ds)}  Valid {len(valid_ds)}  Test {len(test_ds)} functions\n")

    opt = torch.optim.AdamW(model.parameters(), lr=FAE_LR, weight_decay=FAE_WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    ema = EMA(model, FAE_EMA_DECAY)
    eval_model = FunctionAutoencoder(config).to(device)
    frame_gy, frame_gz = _frame_coords(device)

    best_v, best_state, pat = float("inf"), None, 0
    t0 = time.time()
    for ep in range(1, FAE_EPOCHS + 1):
        model.train(); tl = 0.0; nb = 0; agg = {}
        for fields, params, slice_ids, ambient, _ in train_loader:
            fields = fields.to(device, non_blocking=True)
            params = params.to(device, non_blocking=True)
            slice_ids = slice_ids.to(device, non_blocking=True)
            ambient = ambient.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss, parts = fae_step(model, fields, params, slice_ids, ambient,
                                   device, frame_gy, frame_gz)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); ema.update(model)
            tl += loss.item(); nb += 1
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v
        tl /= max(nb, 1)

        if valid_loader is not None:
            eval_model.load_state_dict(ema.state_dict(model))
            vl = validate(eval_model, valid_loader, device, frame_gy, frame_gz)
        else:
            vl = tl

        if ep % 10 == 0 or ep == 1:
            ap = {k: v / max(nb, 1) for k, v in agg.items()}
            print(f"  Ep{ep:4d} T:{tl:.4f} V:{vl:.4f} | "
                  f"mse:{ap['mse']:.4f} ssim:{ap['ssim']:.3f} "
                  f"grow:{ap['growth']:.4f} eng:{ap['energy']:.4f} "
                  f"lr:{sched.get_last_lr()[0]:.2e}")

        if vl < best_v:
            best_v = vl; best_state = ema.state_dict(model); pat = 0
        else:
            pat += 1
            if pat >= FAE_PATIENCE:
                print(f"  Early stop at ep {ep}."); break

    print(f"\n  Best valid MSE: {best_v:.4f}  ({time.time() - t0:.0f}s)")
    model.load_state_dict(best_state)

    ckpt = {
        "fae_state": best_state,
        "norm_stats": norm_stats,
        "latent_shape": model.latent_shape,
        "config": {k: getattr(config, k) for k in [
            "EMBED_DIM", "MLP_WIDTH", "N_HEADS", "ENC_DEPTH", "DEC_DEPTH",
            "N_LATENT_TOKENS", "PATCH_T", "PATCH_H", "PATCH_W",
            "COORD_FOURIER_FEATURES", "COORD_FOURIER_SCALE", "USE_AMBIENT_FLOOR"]},
    }
    torch.save(ckpt, os.path.join(MODEL_DIR, "fae_best.pt"))
    print(f"  Saved -> {os.path.join(MODEL_DIR, 'fae_best.pt')}")

    for name, ds in [("valid", valid_ds), ("test", test_ds)]:
        r2 = full_grid_r2(model, ds, device)
        if r2 is not None:
            print(f"  FAE full-grid reconstruction R2 [{name}]: {r2:.4f}")


if __name__ == "__main__":
    main()
