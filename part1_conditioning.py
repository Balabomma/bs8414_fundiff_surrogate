"""Shared conditioning adapter for the BS 8414 Part1 corpus.

The Part1 parameter vector is

    [cladding_id, insulation_id, geom_id, 13 normalised material features]

Its first three columns are **categorical codes, not magnitudes**. Feeding them
to a model as raw scalars — which every one of the existing slice architectures
would do, since they all take `params` as a flat continuous vector — asserts
that geometry 7 is seven times geometry 1 and that brick sits between aluminium
and cedar on some continuum. On the 60-sim corpus only one column (cladding id)
was categorical and the models that cared embedded it themselves; Part1 has
three, and the two new ones carry the effect under study.

`Part1Conditioner` replaces those three codes with learned embeddings and
concatenates the material block back on, producing a purely continuous vector:

    (B, 16) ->  (B, 49)      12 cladding + 8 insulation + 16 geometry + 13 material

Any encoder that expected a continuous parameter vector then works unchanged —
it just needs building with `n_params = Part1Conditioner.OUT_DIM`. That is what
lets the FNO/Transolver, Samba and diffusion projects adopt the Part1 corpus
without restructuring their architectures.

Geometry is an 8-way embedding over the observed flag combinations (the user's
choice), so the model can learn an arbitrary interaction between removing the
cavity, the gaps and the barriers rather than assuming the three effects are
additive. A combination absent from training has no embedding row.
"""
import torch
import torch.nn as nn

from config_part1 import (
    N_CLADDING, N_INSULATION, N_GEOMETRY, N_MATERIAL_FEATURES,
    CLADDING_EMBED_DIM, INSULATION_EMBED_DIM, GEOMETRY_EMBED_DIM,
    COL_CLADDING, COL_INSULATION, COL_GEOM,
)


class Part1Conditioner(nn.Module):
    """(B, 16) raw Part1 params -> (B, OUT_DIM) continuous conditioning vector."""

    OUT_DIM = (CLADDING_EMBED_DIM + INSULATION_EMBED_DIM + GEOMETRY_EMBED_DIM
               + N_MATERIAL_FEATURES)

    def __init__(self):
        super().__init__()
        self.cladding_embedding = nn.Embedding(N_CLADDING, CLADDING_EMBED_DIM)
        self.insulation_embedding = nn.Embedding(N_INSULATION, INSULATION_EMBED_DIM)
        self.geometry_embedding = nn.Embedding(N_GEOMETRY, GEOMETRY_EMBED_DIM)

    def forward(self, params):
        clad = self.cladding_embedding(
            params[:, COL_CLADDING].long().clamp(0, N_CLADDING - 1))
        ins = self.insulation_embedding(
            params[:, COL_INSULATION].long().clamp(0, N_INSULATION - 1))
        geom = self.geometry_embedding(
            params[:, COL_GEOM].long().clamp(0, N_GEOMETRY - 1))
        return torch.cat([clad, ins, geom, params[:, 3:]], dim=-1)


if __name__ == "__main__":
    from config_part1 import N_INPUT_PARAMS

    cond = Part1Conditioner()
    p = torch.rand(4, N_INPUT_PARAMS)
    p[:, COL_CLADDING] = torch.randint(0, N_CLADDING, (4,)).float()
    p[:, COL_INSULATION] = torch.randint(0, N_INSULATION, (4,)).float()
    p[:, COL_GEOM] = torch.randint(0, N_GEOMETRY, (4,)).float()
    out = cond(p)
    print(f"Part1Conditioner: {tuple(p.shape)} -> {tuple(out.shape)}  "
          f"(OUT_DIM={Part1Conditioner.OUT_DIM})")
    print(f"  parameters: {sum(x.numel() for x in cond.parameters()):,}")
