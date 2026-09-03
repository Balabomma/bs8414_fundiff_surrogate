"""Function Autoencoder (FAE) for FunDiff — Stage 1.

Faithful transplant of the FunDiff FAE (Wang et al., Nat. Commun. 17:5749, 2026),
generalized from 2D fields to the BS 8414 spatiotemporal temperature field
T(t, y, z):

  Encoder E  (resolution-invariant):
    - ViT-style 3D patchify of the (padded) T x H x W field  -> patch tokens
    - factorized, interpolatable positional embeddings (support temporal
      sub-sampling for resolution invariance)
    - a Perceiver cross-attention module mapping the variable-length patch
      sequence onto N fixed learnable latent queries
    - a stack of pre-norm self-attention Transformer blocks
    -> latent tokens z in R^{N x D}   (the "function code")

  Decoder D  (CViT-style, continuous):
    - embed query coordinates (t, y, z) via random Fourier features -> queries
    - K cross-attention Transformer blocks: queries attend to the latent tokens
    - a small FFN -> scalar temperature at each query point
    - OPTIONAL hard ambient floor: out = ambient + softplus(raw), so the
      reconstructed field can never fall below ambient (the analogue of the
      paper's hard divergence-free / periodic decoder modifications).

The autoencoder D . E is trained by reconstructing the field at randomly
sampled continuous (t, y, z) query coordinates (paper eq. for L_FAE).
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Coordinate embedding (random Fourier features) ──────────────────

class FourierCoordEmbedding(nn.Module):
    """(B, N, 3) normalized coords -> (B, N, D) via random Fourier features."""

    def __init__(self, in_dim=3, n_features=64, scale=8.0, out_dim=256):
        super().__init__()
        # Fixed random projection (not learned) — standard RFF.
        B_mat = torch.randn(in_dim, n_features) * scale
        self.register_buffer("B_mat", B_mat)
        self.proj = nn.Linear(2 * n_features, out_dim)

    def forward(self, coords):
        x = 2 * math.pi * coords @ self.B_mat           # (B, N, F)
        feats = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        return self.proj(feats)                         # (B, N, D)


# ── Transformer primitives ──────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, dim, hidden, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, dim),
        )

    def forward(self, x):
        return self.net(x)


class SelfAttnBlock(nn.Module):
    """Pre-norm multi-head self-attention block."""

    def __init__(self, dim, heads, mlp_width, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_width, dropout)

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttnBlock(nn.Module):
    """Pre-norm cross-attention block: `x` (queries) attends to `ctx` (kv)."""

    def __init__(self, dim, heads, mlp_width, dropout=0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_width, dropout)

    def forward(self, x, ctx):
        q = self.norm_q(x)
        kv = self.norm_kv(ctx)
        x = x + self.attn(q, kv, kv, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


# ── Encoder ──────────────────────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(self, grid_shape, patch, embed_dim=256, mlp_width=512,
                 heads=8, depth=8, n_latent=128, dropout=0.0):
        super().__init__()
        self.grid_shape = grid_shape          # base (T, H, W) BEFORE temporal pad
        self.pt, self.ph, self.pw = patch
        T, H, W = grid_shape
        # pad T up to a multiple of pt (H, W assumed divisible)
        self.T_pad = int(math.ceil(T / self.pt) * self.pt)
        self.nt = self.T_pad // self.pt
        self.nh = H // self.ph
        self.nw = W // self.pw
        assert H % self.ph == 0 and W % self.pw == 0, "H,W must divide the patch size"

        self.patchify = nn.Conv3d(1, embed_dim, kernel_size=patch, stride=patch)

        # Factorized, interpolatable positional embeddings.
        self.pe_t = nn.Parameter(torch.zeros(1, self.nt, embed_dim))
        self.pe_s = nn.Parameter(torch.zeros(1, self.nh * self.nw, embed_dim))
        nn.init.trunc_normal_(self.pe_t, std=0.02)
        nn.init.trunc_normal_(self.pe_s, std=0.02)

        # Perceiver latent queries.
        self.latents = nn.Parameter(torch.zeros(1, n_latent, embed_dim))
        nn.init.trunc_normal_(self.latents, std=0.02)
        self.perceiver = CrossAttnBlock(embed_dim, heads, mlp_width, dropout)

        self.blocks = nn.ModuleList(
            [SelfAttnBlock(embed_dim, heads, mlp_width, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim

    def _add_pe(self, tokens, nt):
        """tokens: (B, nt, nh*nw, D)."""
        pe_t = self.pe_t
        if nt != self.nt:  # temporal sub-sampling changed the temporal length
            pe_t = F.interpolate(
                self.pe_t.transpose(1, 2), size=nt, mode="linear",
                align_corners=False).transpose(1, 2)
        pe = pe_t.unsqueeze(2) + self.pe_s.unsqueeze(1)   # (1, nt, nh*nw, D)
        return tokens + pe

    def forward(self, field):
        """field: (B, T, H, W) -> latent tokens (B, n_latent, D).

        Resolution-invariant in time: the input is assumed to span the SAME
        temporal extent [0, 1] regardless of its frame count T (e.g. a
        stride-`factor` sub-sample of the full field).  We pad only up to the
        next multiple of the temporal patch size and interpolate the temporal
        positional embedding to the resulting token count, so the encoder always
        sees a consistent parameterization of the function over [0, 1] and the
        decoder's real (t, y, z) queries stay aligned.
        """
        B, T, H, W = field.shape
        x = field.unsqueeze(1)                            # (B, 1, T, H, W)
        pad_T = int(math.ceil(T / self.pt) * self.pt)
        if pad_T > T:
            x = F.pad(x, (0, 0, 0, 0, 0, pad_T - T))      # pad time at the end

        p = self.patchify(x)                              # (B, D, nt, nh, nw)
        nt = p.shape[2]
        tokens = p.flatten(3).permute(0, 2, 3, 1)         # (B, nt, nh*nw, D)
        tokens = self._add_pe(tokens, nt)
        seq = tokens.reshape(B, -1, self.embed_dim)       # (B, nt*nh*nw, D)

        lat = self.latents.expand(B, -1, -1)              # (B, N, D)
        lat = self.perceiver(lat, seq)                    # cross-attn to patches
        for blk in self.blocks:
            lat = blk(lat)
        return self.norm(lat)                             # (B, N, D)


# ── Decoder (CViT continuous) ────────────────────────────────────────

class Decoder(nn.Module):
    def __init__(self, embed_dim=256, mlp_width=512, heads=8, depth=4,
                 n_fourier=64, fourier_scale=8.0, dropout=0.0, use_ambient_floor=True):
        super().__init__()
        self.coord_embed = FourierCoordEmbedding(3, n_fourier, fourier_scale, embed_dim)
        self.blocks = nn.ModuleList(
            [CrossAttnBlock(embed_dim, heads, mlp_width, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, 1))
        self.use_ambient_floor = use_ambient_floor

    def forward(self, latent, coords, ambient_norm=None):
        """latent: (B, N, D); coords: (B, Q, 3) -> (B, Q) temperature.

        ambient_norm: (B,) normalized ambient — enforces a hard floor when
        use_ambient_floor is set.
        """
        x = self.coord_embed(coords)                      # (B, Q, D)
        for blk in self.blocks:
            x = blk(x, latent)
        raw = self.head(self.norm(x)).squeeze(-1)         # (B, Q)
        if self.use_ambient_floor and ambient_norm is not None:
            return ambient_norm.unsqueeze(1) + F.softplus(raw)
        return raw


# ── Full autoencoder ─────────────────────────────────────────────────

class FunctionAutoencoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(
            grid_shape=config.GRID_SHAPE,
            patch=(config.PATCH_T, config.PATCH_H, config.PATCH_W),
            embed_dim=config.EMBED_DIM, mlp_width=config.MLP_WIDTH,
            heads=config.N_HEADS, depth=config.ENC_DEPTH,
            n_latent=config.N_LATENT_TOKENS, dropout=config.FAE_DROPOUT,
        )
        self.decoder = Decoder(
            embed_dim=config.EMBED_DIM, mlp_width=config.MLP_WIDTH,
            heads=config.N_HEADS, depth=config.DEC_DEPTH,
            n_fourier=config.COORD_FOURIER_FEATURES,
            fourier_scale=config.COORD_FOURIER_SCALE,
            dropout=config.FAE_DROPOUT,
            use_ambient_floor=config.USE_AMBIENT_FLOOR,
        )
        self.latent_shape = (config.N_LATENT_TOKENS, config.EMBED_DIM)

    def encode(self, field):
        return self.encoder(field)

    def decode(self, latent, coords, ambient_norm=None):
        return self.decoder(latent, coords, ambient_norm)

    def forward(self, field, coords, ambient_norm=None):
        return self.decode(self.encode(field), coords, ambient_norm)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config
    m = FunctionAutoencoder(config)
    print(f"FAE params: {count_parameters(m):,}")
    field = torch.randn(2, config.N_TIMESTEPS, config.FIELD_HEIGHT, config.FIELD_WIDTH)
    coords = torch.rand(2, 512, 3)
    amb = torch.tensor([-0.6, -0.6])
    out = m(field, coords, amb)
    print(f"field {tuple(field.shape)} -> latent {tuple(m.encode(field).shape)} "
          f"-> decode {tuple(out.shape)}  min={out.min():.3f}")
