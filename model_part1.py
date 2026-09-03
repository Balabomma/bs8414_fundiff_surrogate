"""FunDiff DiT conditioning for the BS 8414 Part1 geometry-variant corpus.

Parent: `model/dit.py` (`DiT`, `ParamConditioner`) — the rectified-flow velocity
model over FAE latent tokens. Only the conditioning path changes; the DiT
blocks, the AdaLN-Zero modulation, the rectified-flow objective and the Stage-1
Function Autoencoder are untouched.

Why the parent's conditioner is wrong for Part1: `ParamConditioner` concatenates
the raw 16-d parameter vector with a slice embedding and pushes it through an
MLP. On the 60-sim corpus 15 of those 16 columns were genuine continuous
quantities. Under Part1 the first three are categorical codes — cladding,
insulation and geometry — and feeding a geometry id of 7 into a Linear asserts
it is seven times geometry 1. Since geometry is the variable this corpus exists
to study, that is not a detail. `Part1Conditioner` embeds all three and hands
the same MLP a purely continuous 49-d vector.

The Stage-1 FAE needs no change: it autoencodes the field itself and never sees
the parameter vector.

To train on Part1, each stage needs its data source swapped — one import line:

    train_fae.py, train_dit.py:
        from data_processing.dataset import build_datasets, collate
     -> from dataset_part1        import build_datasets, collate

and Stage 2 additionally builds `Part1DiT(config)` in place of `DiT(config)`.

Run `python model_part1.py` for the parameter count and a forward-pass check.
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as parent_config
from part1_conditioning import Part1Conditioner
from model.dit import DiT, rectified_flow_loss, sample_latent, count_parameters


class Part1ParamConditioner(nn.Module):
    """Part1 conditioning + slice embedding -> (B, D). Drop-in for the parent's."""

    def __init__(self, n_slices, slice_embed_dim, out_dim):
        super().__init__()
        self.conditioner = Part1Conditioner()
        self.slice_embed = nn.Embedding(n_slices, slice_embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(Part1Conditioner.OUT_DIM + slice_embed_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, params, slice_ids):
        return self.mlp(torch.cat(
            [self.conditioner(params), self.slice_embed(slice_ids)], dim=-1))


class Part1DiT(DiT):
    """Parent DiT with the Part1 conditioner swapped in.

    Subclassed rather than copied so the transformer blocks and the
    rectified-flow head under test stay provably identical to the parent's.
    """

    def __init__(self, config=parent_config):
        # Default so the shared trainer's `Part1Surrogate()` call works; the
        # explicit form `Part1Surrogate(cfg)` is unchanged.
        super().__init__(config)
        self.cond = Part1ParamConditioner(
            config.N_SLICES, config.SLICE_EMBED_DIM, config.DIT_EMBED_DIM)


# ── uniform interface (see bs8414_KAN_surrogate/model_part1.py) ───────────
MODEL_NAME = "FunDiff DiT (Part1)"
Part1Surrogate = Part1DiT
LAMBDA_REG = 0.0  # rectified flow carries no extra weight penalty


def regularization(model):
    return torch.zeros((), device=next(model.parameters()).device)


if __name__ == "__main__":
    from config_part1 import (N_INPUT_PARAMS, N_CLADDING, N_INSULATION,
                              N_GEOMETRY, COL_CLADDING, COL_INSULATION, COL_GEOM)

    model = Part1Surrogate(parent_config)
    print(f"{MODEL_NAME}: {count_parameters(model):,} parameters")

    B = 4
    p = torch.rand(B, N_INPUT_PARAMS)
    p[:, COL_CLADDING] = torch.randint(0, N_CLADDING, (B,)).float()
    p[:, COL_INSULATION] = torch.randint(0, N_INSULATION, (B,)).float()
    p[:, COL_GEOM] = torch.randint(0, N_GEOMETRY, (B,)).float()
    slice_ids = torch.randint(0, parent_config.N_SLICES, (B,))

    z = torch.randn(B, parent_config.N_LATENT_TOKENS, parent_config.EMBED_DIM)
    t = torch.rand(B)
    v = model(z, t, p, slice_ids)
    print(f"  latent {tuple(z.shape)} -> velocity {tuple(v.shape)}")

    loss = rectified_flow_loss(model, z, p, slice_ids)
    loss = loss[0] if isinstance(loss, tuple) else loss
    print(f"  rectified-flow loss on random latents: {float(loss):.4f}")
