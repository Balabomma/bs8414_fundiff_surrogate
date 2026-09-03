r"""Stage 2 — train the latent Diffusion Transformer (DiT).

Freezes the Stage-1 FAE, encodes every training function once into a latent
z = E(f) in R^{N x D}, and trains the DiT with rectified flow (paper eq. 9) to
generate those latents conditioned on the 16-d parameter vector + slice id.

The latents are tiny (~145 functions x 128 x 256), so they are precomputed once
and cached on the GPU; training is fast and uses a large batch.

Run:  venv\Scripts\activate  &&  python train_dit.py   (requires fae_best.pt)
"""
import os
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from config import (
    MODEL_DIR, DEVICE, EMBED_DIM, N_LATENT_TOKENS,
    DIT_LR, DIT_WARMUP_STEPS, DIT_DECAY_EVERY, DIT_DECAY_RATE, DIT_WEIGHT_DECAY,
    DIT_BATCH_SIZE, DIT_EPOCHS, DIT_PATIENCE, DIT_EMA_DECAY,
)
from dataset_part1 import build_datasets, collate
from model.fae import FunctionAutoencoder
from model.dit import rectified_flow_loss, count_parameters
# Uniform Part1 interface: Part1DiT here, Part1KANDiT in the KAN variant.
from model_part1 import Part1Surrogate, MODEL_NAME
from train_fae import EMA


def dit_lr_lambda(step):
    if step < DIT_WARMUP_STEPS:
        return (step + 1) / DIT_WARMUP_STEPS
    return DIT_DECAY_RATE ** ((step - DIT_WARMUP_STEPS) / DIT_DECAY_EVERY)


@torch.no_grad()
def encode_split(fae, dataset, device):
    """Encode every function -> (latents, params, slice_ids)."""
    fae.eval()
    lats, ps, ss = [], [], []
    loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate)
    for fields, params, slice_ids, ambient, _ in loader:
        z = fae.encode(fields.to(device))
        lats.append(z.cpu()); ps.append(params); ss.append(slice_ids)
    if not lats:
        return None
    return (torch.cat(lats), torch.cat(ps), torch.cat(ss))


def main():
    print("=" * 70)
    print("  BS8414 FunDiff surrogate — Stage 2: latent DiT (rectified flow)")
    print("=" * 70)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    fae_path = os.path.join(MODEL_DIR, "fae_best.pt")
    if not os.path.isfile(fae_path):
        print("ERROR: fae_best.pt not found — run train_fae.py first."); return

    ckpt = torch.load(fae_path, map_location=device, weights_only=False)
    fae = FunctionAutoencoder(config).to(device)
    fae.load_state_dict(ckpt["fae_state"])
    for p in fae.parameters():
        p.requires_grad_(False)

    train_ds, valid_ds, _, norm_stats = build_datasets()
    train = encode_split(fae, train_ds, device)
    valid = encode_split(fae, valid_ds, device)
    if train is None:
        print("ERROR: no training latents."); return

    z_tr, p_tr, s_tr = train
    print(f"  Train latents: {tuple(z_tr.shape)}  (mu={z_tr.mean():.3f} sd={z_tr.std():.3f})")
    tr_loader = DataLoader(TensorDataset(z_tr, p_tr, s_tr),
                           batch_size=DIT_BATCH_SIZE, shuffle=True)

    # Seed is env-overridable so independent replicates can be trained without
    # editing the tracked default of 42.
    _seed = int(__import__('os').environ.get('FUNDIFF_SEED', 42))
    torch.manual_seed(_seed); np.random.seed(_seed)
    print(f'  seed: {_seed}')
    model = Part1Surrogate(config).to(device)
    print(f"  DiT params: {count_parameters(model):,}\n")
    opt = torch.optim.AdamW(model.parameters(), lr=DIT_LR, weight_decay=DIT_WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, dit_lr_lambda)
    ema = EMA(model, DIT_EMA_DECAY)
    eval_model = Part1Surrogate(config).to(device)

    best_v, best_state, pat = float("inf"), None, 0
    t0 = time.time()
    for ep in range(1, DIT_EPOCHS + 1):
        model.train(); tl, nb = 0.0, 0
        for z, p, s in tr_loader:
            z = z.to(device); p = p.to(device); s = s.to(device)
            opt.zero_grad(set_to_none=True)
            loss = rectified_flow_loss(model, z, p, s)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); ema.update(model)
            tl += loss.item(); nb += 1
        tl /= max(nb, 1)

        # validation: EMA rectified-flow loss on valid latents (fixed seed)
        if valid is not None:
            eval_model.load_state_dict(ema.state_dict(model)); eval_model.eval()
            g = torch.Generator(device=device).manual_seed(0)
            zv, pv, sv = valid
            with torch.no_grad():
                vl = rectified_flow_loss(
                    eval_model, zv.to(device), pv.to(device), sv.to(device), generator=g).item()
        else:
            vl = tl

        if ep % 50 == 0 or ep == 1:
            print(f"  Ep{ep:4d}  T:{tl:.4f}  V(ema):{vl:.4f}  lr:{sched.get_last_lr()[0]:.2e}")

        if vl < best_v:
            best_v = vl; best_state = ema.state_dict(model); pat = 0
        else:
            pat += 1
            if pat >= DIT_PATIENCE:
                print(f"  Early stop at ep {ep}."); break

    print(f"\n  Best valid rf-loss: {best_v:.4f}  ({time.time() - t0:.0f}s)")
    torch.save({
        "dit_state": best_state,
        "fae_path": "fae_best.pt",
        "latent_shape": (N_LATENT_TOKENS, EMBED_DIM),
        "norm_stats": norm_stats,
    }, os.path.join(MODEL_DIR, "dit_best.pt"))
    print(f"  Saved -> {os.path.join(MODEL_DIR, 'dit_best.pt')}")


if __name__ == "__main__":
    main()
