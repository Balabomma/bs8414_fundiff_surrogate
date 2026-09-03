"""Latent Diffusion Transformer (DiT) for FunDiff — Stage 2.

Operates on the frozen FAE latent tokens z in R^{N x D}.  Follows the paper:
  - Diffusion Transformer backbone with AdaLN-Zero modulation (Peebles & Xie,
    ref. 76), scaling factors init to zero so each block starts as identity.
  - Trained with RECTIFIED FLOW (paper eq. 9).  We use the standard
    noise(t=0) -> data(t=1) parameterization:
        x0 ~ N(0, I),  x1 = E(f)  (data latent)
        x_t = (1 - t) x0 + t x1,     t ~ U[0, 1]
        target velocity  v = x1 - x0
    This matches paper eq. (9) up to the relabeling tau = 1 - t.
  - CONDITIONING: unlike the paper's coarse-field conditioning (added to the
    latent), the BS 8414 conditioning is the 16-d parameter vector + slice id.
    We inject it through AdaLN alongside the timestep embedding — the native
    conditioning path of the DiT the paper builds on.

Inference integrates dz/dt = v(z, t, cond) from t=0 (Gaussian noise) to t=1
(data latent) with a fixed-step Euler / midpoint ODE solver (20-50 steps).
"""
import math
import torch
import torch.nn as nn


def timestep_embedding(t, dim, max_period=10000):
    """Sinusoidal embedding of a scalar t in [0, 1]. t: (B,) -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device).float() / half)
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0) * 1000.0
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class ParamConditioner(nn.Module):
    """16-d params + slice-id embedding -> conditioning vector (B, D)."""

    def __init__(self, n_params, n_slices, slice_embed_dim, out_dim):
        super().__init__()
        self.slice_embed = nn.Embedding(n_slices, slice_embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(n_params + slice_embed_dim, out_dim), nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, params, slice_ids):
        s = self.slice_embed(slice_ids)
        return self.mlp(torch.cat([params, s], dim=-1))


class DiTBlock(nn.Module):
    """Transformer block with AdaLN-Zero modulation."""

    def __init__(self, dim, heads, mlp_width, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_width), nn.GELU(), nn.Linear(mlp_width, dim))
        # AdaLN-Zero: produce 6 modulation tensors from the conditioning vector.
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    @staticmethod
    def _modulate(x, shift, scale):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x, c):
        sh_msa, sc_msa, g_msa, sh_mlp, sc_mlp, g_mlp = self.ada(c).chunk(6, dim=-1)
        h = self._modulate(self.norm1(x), sh_msa, sc_msa)
        x = x + g_msa.unsqueeze(1) * self.attn(h, h, h, need_weights=False)[0]
        h = self._modulate(self.norm2(x), sh_mlp, sc_mlp)
        x = x + g_mlp.unsqueeze(1) * self.mlp(h)
        return x


class DiT(nn.Module):
    """Rectified-flow velocity model over FAE latent tokens."""

    def __init__(self, config):
        super().__init__()
        D = config.DIT_EMBED_DIM
        N = config.N_LATENT_TOKENS
        self.n_tokens = N
        self.latent_dim = config.EMBED_DIM
        assert config.EMBED_DIM == D, "DiT operates directly on FAE latent width"

        self.pos = nn.Parameter(torch.zeros(1, N, D))
        nn.init.trunc_normal_(self.pos, std=0.02)

        self.t_mlp = nn.Sequential(nn.Linear(D, D), nn.SiLU(), nn.Linear(D, D))
        self.cond = ParamConditioner(
            config.N_INPUT_PARAMS, config.N_SLICES,
            config.SLICE_EMBED_DIM, D)

        self.blocks = nn.ModuleList([
            DiTBlock(D, config.DIT_HEADS, config.DIT_MLP_WIDTH, config.DIT_DROPOUT)
            for _ in range(config.DIT_DEPTH)])

        self.norm_out = nn.LayerNorm(D, elementwise_affine=False, eps=1e-6)
        self.ada_out = nn.Sequential(nn.SiLU(), nn.Linear(D, 2 * D))
        nn.init.zeros_(self.ada_out[-1].weight)
        nn.init.zeros_(self.ada_out[-1].bias)
        self.head = nn.Linear(D, D)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.embed_dim = D

    def _cond_vec(self, t, params, slice_ids):
        temb = self.t_mlp(timestep_embedding(t, self.embed_dim))
        return temb + self.cond(params, slice_ids)        # (B, D)

    def forward(self, x, t, params, slice_ids):
        """x: (B, N, D) noisy latent; t: (B,) -> velocity (B, N, D)."""
        c = self._cond_vec(t, params, slice_ids)
        x = x + self.pos
        for blk in self.blocks:
            x = blk(x, c)
        sh, sc = self.ada_out(c).chunk(2, dim=-1)
        x = self.norm_out(x) * (1 + sc.unsqueeze(1)) + sh.unsqueeze(1)
        return self.head(x)


# ── Rectified-flow helpers ───────────────────────────────────────────

def rectified_flow_loss(model, x_data, params, slice_ids, generator=None):
    """Rectified-flow training loss (paper eq. 9), x0=noise, x1=data."""
    B = x_data.shape[0]
    device = x_data.device
    t = torch.rand(B, device=device, generator=generator)
    x0 = torch.randn(x_data.shape, device=device, generator=generator)
    tt = t.view(B, 1, 1)
    x_t = (1 - tt) * x0 + tt * x_data
    v_target = x_data - x0
    v_pred = model(x_t, t, params, slice_ids)
    return ((v_pred - v_target) ** 2).mean()


@torch.no_grad()
def sample_latent(model, params, slice_ids, n_tokens, latent_dim,
                  n_steps=50, generator=None, x0=None):
    """Integrate dz/dt = v from t=0 (noise) to t=1 (data) with midpoint Euler."""
    B = params.shape[0]
    device = params.device
    if x0 is None:
        x0 = torch.randn(B, n_tokens, latent_dim, device=device, generator=generator)
    x = x0
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((B,), i * dt, device=device)
        v1 = model(x, t, params, slice_ids)
        # midpoint (RK2) for accuracy at low step counts
        t_mid = torch.full((B,), (i + 0.5) * dt, device=device)
        v2 = model(x + 0.5 * dt * v1, t_mid, params, slice_ids)
        x = x + dt * v2
    return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config
    m = DiT(config)
    print(f"DiT params: {count_parameters(m):,}")
    x = torch.randn(4, config.N_LATENT_TOKENS, config.EMBED_DIM)
    p = torch.rand(4, config.N_INPUT_PARAMS)
    s = torch.randint(0, config.N_SLICES, (4,))
    loss = rectified_flow_loss(m, x, p, s)
    print(f"rf loss: {loss.item():.4f}")
    z = sample_latent(m, p, s, config.N_LATENT_TOKENS, config.EMBED_DIM, n_steps=10)
    print(f"sampled latent: {tuple(z.shape)}")
