"""Dataset for the FunDiff spatiotemporal function surrogate.

Unlike the other slice surrogates (which yield short temporal windows), one
sample here is the WHOLE spatiotemporal temperature field for a single
(simulation, slice):

    field : (T, H, W)   standardized temperature (per-slice mean/std)
    params: (16,)       normalized parameter vector
    slice_id : int      slice index [0..4]
    ambient_norm : float   (T_ambient - mean_slice) / std_slice  (for the floor)

The encoder consumes the full padded field; the FAE reconstruction loss and the
DiT are trained by sampling random continuous (t, y, z) query coordinates from
the true 181x128x64 grid (see sample_query_points()).  This mirrors the paper's
protocol: "each training batch contains ... 4096 query coordinates and their
corresponding outputs randomly sampled from the grid".

Split is the IDENTICAL deterministic md5-hash-on-chid scheme used by every other
surrogate, so train/valid/test membership matches for cross-model comparison.
"""
import os
import sys
import json
import hashlib
import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    NUMPY_DIR, METADATA_PATH, NORM_STATS_PATH,
    SLICE_IDS, SLICE_ID_MAP, FIELD_HEIGHT, FIELD_WIDTH, N_TIMESTEPS,
    N_INPUT_PARAMS, T_AMBIENT, TRAIN_RATIO, VALID_RATIO,
)


def assign_split(chid, train_ratio=TRAIN_RATIO, valid_ratio=VALID_RATIO):
    """Deterministic md5-hash split on the chid — identical across all surrogates."""
    h = hashlib.md5(chid.encode()).hexdigest()
    bucket = int(h[:8], 16) / 0xFFFFFFFF
    if bucket < train_ratio:
        return "train"
    elif bucket < train_ratio + valid_ratio:
        return "valid"
    return "test"


class FunctionFieldDataset(Dataset):
    """Yields whole spatiotemporal fields T(t, y, z) for (sim, slice) pairs."""

    def __init__(self, metadata, norm_stats, split="train"):
        self.split = split
        self.norm_stats = norm_stats
        self.samples = []          # list of (chid, slice_id)
        self.fields = {}           # (chid, slice_id) -> (T, H, W) float32 normalized
        self.params = {}           # (chid, slice_id) -> (16,) float32
        self.ambient = {}          # (chid, slice_id) -> float  (normalized ambient)

        n_loaded = 0
        for chid, meta in metadata.items():
            if assign_split(chid) != split:
                continue
            if chid not in set(os.listdir(NUMPY_DIR)):
                continue
            params = np.asarray(meta["params"], dtype=np.float32)

            for slice_id in SLICE_IDS:
                if slice_id not in meta.get("slices", {}):
                    continue
                slice_dir = os.path.join(NUMPY_DIR, chid, slice_id)
                t_indices = meta["slices"][slice_id]["timestep_indices"]

                frames = []
                for t_idx in t_indices:
                    fpath = os.path.join(slice_dir, f"t{t_idx:04d}.npy")
                    if os.path.isfile(fpath):
                        frames.append(np.load(fpath).astype(np.float32))
                if len(frames) != N_TIMESTEPS:
                    # require the full temporal grid so (t) coords are well-defined
                    continue

                field = np.stack(frames)  # (T, H, W) in physical deg C
                stats = norm_stats.get(slice_id, norm_stats["global"])
                mean, std = stats["mean"], stats["std"]
                field_norm = (field - mean) / std

                key = (chid, slice_id)
                self.fields[key] = field_norm
                self.params[key] = params
                self.ambient[key] = (T_AMBIENT - mean) / std
                self.samples.append(key)
                n_loaded += 1

        mem_gb = n_loaded * N_TIMESTEPS * FIELD_HEIGHT * FIELD_WIDTH * 4 / 1e9
        print(f"  [{split}] {n_loaded} functions "
              f"({len({c for c, _ in self.samples})} sims, {mem_gb:.2f} GB in RAM)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        key = self.samples[idx]
        field = self.fields[key]                       # (T, H, W)
        params = self.params[key]                      # (16,)
        slice_idx = SLICE_ID_MAP[key[1]]
        return (
            torch.from_numpy(field),                   # (T, H, W)
            torch.from_numpy(params),                  # (16,)
            torch.tensor(slice_idx, dtype=torch.long),
            torch.tensor(self.ambient[key], dtype=torch.float32),
            key[0],                                    # chid (for eval bookkeeping)
        )


def collate(batch):
    """Stack fields; keep chids as a python list."""
    fields = torch.stack([b[0] for b in batch])        # (B, T, H, W)
    params = torch.stack([b[1] for b in batch])        # (B, 16)
    slice_ids = torch.stack([b[2] for b in batch])     # (B,)
    ambient = torch.stack([b[3] for b in batch])       # (B,)
    chids = [b[4] for b in batch]
    return fields, params, slice_ids, ambient, chids


# ── Query-coordinate sampling ────────────────────────────────────────

def sample_query_points(fields, n_query, device, generator=None):
    """Randomly sample n_query (t, y, z) points per function and gather targets.

    fields: (B, T, H, W) standardized temperature.
    Returns:
        coords : (B, n_query, 3)  normalized (t, y, z) in [0, 1]
        target : (B, n_query)     standardized temperature at those points
    """
    B, T, H, W = fields.shape
    ti = torch.randint(0, T, (B, n_query), device=device, generator=generator)
    yi = torch.randint(0, H, (B, n_query), device=device, generator=generator)
    zi = torch.randint(0, W, (B, n_query), device=device, generator=generator)

    coords = torch.stack([
        ti.float() / max(T - 1, 1),
        yi.float() / max(H - 1, 1),
        zi.float() / max(W - 1, 1),
    ], dim=-1)  # (B, n_query, 3)

    bidx = torch.arange(B, device=device).unsqueeze(1).expand(-1, n_query)
    target = fields[bidx, ti, yi, zi]  # (B, n_query)
    return coords, target


def full_grid_coords(T=N_TIMESTEPS, H=FIELD_HEIGHT, W=FIELD_WIDTH, device="cpu"):
    """Dense (T*H*W, 3) normalized coordinate grid for full-field decoding.

    Supports super-resolution: pass a larger T/H/W than the native grid to
    evaluate the continuous decoder off the training discretization.
    """
    tt = torch.linspace(0, 1, T, device=device)
    yy = torch.linspace(0, 1, H, device=device)
    zz = torch.linspace(0, 1, W, device=device)
    gt, gy, gz = torch.meshgrid(tt, yy, zz, indexing="ij")
    return torch.stack([gt.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=-1)


def build_datasets():
    if not os.path.isfile(METADATA_PATH):
        raise FileNotFoundError(f"metadata.json not found at {METADATA_PATH}")
    if not os.path.isfile(NORM_STATS_PATH):
        raise FileNotFoundError(f"norm_stats.json not found at {NORM_STATS_PATH}")

    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    with open(NORM_STATS_PATH) as f:
        norm_stats = json.load(f)

    train = FunctionFieldDataset(metadata, norm_stats, "train")
    valid = FunctionFieldDataset(metadata, norm_stats, "valid")
    test = FunctionFieldDataset(metadata, norm_stats, "test")
    return train, valid, test, norm_stats


if __name__ == "__main__":
    tr, va, te, stats = build_datasets()
    print(f"\nTrain {len(tr)}  Valid {len(va)}  Test {len(te)} functions")
    if len(tr):
        f, p, s, amb, chid = tr[0]
        print(f"field={tuple(f.shape)} params={tuple(p.shape)} "
              f"slice={s.item()} ambient_norm={amb.item():.3f} chid={chid}")
        c, t = sample_query_points(f.unsqueeze(0), 16, "cpu")
        print(f"query coords={tuple(c.shape)} target={tuple(t.shape)} "
              f"t-range=[{t.min():.2f},{t.max():.2f}]")
