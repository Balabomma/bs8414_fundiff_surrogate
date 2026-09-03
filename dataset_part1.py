"""Part1 corpus adapter for the FunDiff two-stage pipeline.

Exposes `build_datasets()` and `collate()` with the SAME names, signatures and
return contract as `data_processing/dataset.py`, so the existing Stage-1 and
Stage-2 trainers move to the Part1 corpus by changing one import line each:

    train_fae.py :  from data_processing.dataset import build_datasets, collate
    train_dit.py :  from data_processing.dataset import build_datasets, collate
                 -> from dataset_part1 import build_datasets, collate

Nothing else in those scripts needs to change; the rectified-flow objective, the
FAE and the DiT training loops are untouched.

Two substantive differences from the parent dataset, both forced by the corpus:

  1. **Source layout.** Reads the packed `.npz` written by
     `bs8414_slice_surrogate/extract_slices_part1.py` rather than a directory of
     loose per-timestep `.npy` frames.

  2. **Absent planes.** The 92 `noair` cases have no ventilated cavity, so
     `Wing_cavity` and `Main_cavity` do not exist for them and are simply not
     emitted as samples. The parent skipped a slice when the metadata lacked it,
     which has the same effect — but here the absence is physical and expected
     rather than a gap, so it is counted and reported instead of passing silently.

Normalisation follows the parent: per-slice mean/std where available, with a
global fallback, fitted on the TRAINING split only.
"""
import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_part1 import (
    SLICE_IDS, SLICE_ID_MAP, N_SLICES, N_TIMESTEPS, FIELD_HEIGHT, FIELD_WIDTH,
    T_AMBIENT, GEOMETRY_NAMES,
)
from slice_loader_part1 import load_corpus
from data_loader_part1 import assign_split


def _fit_norm_stats(fields, time_mask, plane_mask, train_idx):
    """Per-slice and global mean/std over TRAIN rows and existing planes only."""
    stats = {}
    all_values = []
    for p, sid in enumerate(SLICE_IDS):
        chunks = []
        for i in train_idx:
            if plane_mask[i, p] == 0:
                continue
            steps = time_mask[i, p] > 0
            if steps.any():
                chunks.append(fields[i, p][steps].astype(np.float32).ravel())
        if chunks:
            block = np.concatenate(chunks)
            stats[sid] = {"mean": float(block.mean()),
                          "std": float(max(block.std(), 1e-6))}
            all_values.append(block)
    block = np.concatenate(all_values)
    stats["global"] = {"mean": float(block.mean()),
                       "std": float(max(block.std(), 1e-6))}
    return stats


class Part1FunctionFieldDataset(Dataset):
    """Yields whole spatiotemporal fields T(t, y, z) for (sim, plane) pairs."""

    def __init__(self, fields, time_mask, plane_mask, params, meta, indices,
                 norm_stats, split="train"):
        self.split = split
        self.norm_stats = norm_stats
        self.samples = []
        self.fields = {}
        self.params = {}
        self.ambient = {}

        n_skipped_partial = 0
        for i in indices:
            chid = meta[i]["chid"]
            for p, sid in enumerate(SLICE_IDS):
                if plane_mask[i, p] == 0:
                    continue                       # plane does not exist: no cavity
                if (time_mask[i, p] > 0).sum() != N_TIMESTEPS:
                    n_skipped_partial += 1         # full grid required for (t) coords
                    continue

                stats = norm_stats.get(sid, norm_stats["global"])
                mean, std = stats["mean"], stats["std"]

                key = (chid, sid)
                self.fields[key] = ((fields[i, p].astype(np.float32) - mean) / std)
                self.params[key] = params[i].astype(np.float32)
                self.ambient[key] = (T_AMBIENT - mean) / std
                self.samples.append(key)

        n_sims = len({c for c, _ in self.samples})
        mem_gb = len(self.samples) * N_TIMESTEPS * FIELD_HEIGHT * FIELD_WIDTH * 4 / 1e9
        note = f", {n_skipped_partial} skipped (incomplete time grid)" \
            if n_skipped_partial else ""
        print(f"  [{split}] {len(self.samples)} functions "
              f"({n_sims} sims, {mem_gb:.2f} GB in RAM){note}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        key = self.samples[idx]
        return (
            torch.from_numpy(self.fields[key]),
            torch.from_numpy(self.params[key]),
            torch.tensor(SLICE_ID_MAP[key[1]], dtype=torch.long),
            torch.tensor(self.ambient[key], dtype=torch.float32),
            key[0],
        )


def collate(batch):
    """Stack fields; keep chids as a python list. Same contract as the parent."""
    return (torch.stack([b[0] for b in batch]),
            torch.stack([b[1] for b in batch]),
            torch.stack([b[2] for b in batch]),
            torch.stack([b[3] for b in batch]),
            [b[4] for b in batch])


def build_datasets():
    """Returns (train, valid, test, norm_stats) — the parent's contract."""
    params, fields, time_mask, plane_mask, meta = load_corpus(verbose=True)

    idx = {"train": [], "valid": [], "test": []}
    for i, m in enumerate(meta):
        idx[assign_split(m)].append(i)

    norm_stats = _fit_norm_stats(fields, time_mask, plane_mask, idx["train"])
    print(f"  norm stats fitted on {len(idx['train'])} training sims "
          f"(global mean {norm_stats['global']['mean']:.1f} degC, "
          f"sd {norm_stats['global']['std']:.1f})")

    splits = [Part1FunctionFieldDataset(fields, time_mask, plane_mask, params,
                                        meta, idx[name], norm_stats, name)
              for name in ("train", "valid", "test")]
    return splits[0], splits[1], splits[2], norm_stats


if __name__ == "__main__":
    train, valid, test, stats = build_datasets()
    print("\n  per-slice normalisation:")
    for sid in SLICE_IDS:
        s = stats.get(sid)
        print(f"    {sid:<16} " + (f"mean {s['mean']:>7.1f}  sd {s['std']:>7.1f}"
                                   if s else "absent from the training split"))

    field, p, sl, amb, chid = train[0]
    print(f"\n  sample: field {tuple(field.shape)}  params {tuple(p.shape)}  "
          f"slice {int(sl)} ({SLICE_IDS[int(sl)]})  ambient {float(amb):.3f}")
    print(f"          chid {chid}")
    fields, params, slice_ids, ambient, chids = collate([train[i] for i in range(4)])
    print(f"  batch:  fields {tuple(fields.shape)}  params {tuple(params.shape)}  "
          f"slice_ids {tuple(slice_ids.shape)}")
