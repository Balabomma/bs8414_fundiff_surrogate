"""Physics priors for the FunDiff FAE decoder.

Enforced in CONTINUOUS space on the decoder output — the FunDiff design point:
because the decoder is a continuous neural representation, temporal derivatives
and field statistics are evaluated directly on decoded query points, decoupled
from the diffusion stage and free of discretization artefacts.

Two soft priors (the hard ambient floor lives in the decoder itself):

  growth_monotonicity_loss
      During the growth phase (normalized t < GROWTH_PHASE_FRAC) the facade
      temperature field is expected non-decreasing in time.  We decode each
      sampled point at t and at t+dt and penalize any decrease.

  energy_hrr_loss
      The spatial-mean temperature should increase with the prescribed HRR.
      Penalizes a negative batch correlation between mean decoded T and the
      (normalized) HRR parameter — the same "energy/HRR" sanity constraint used
      by the other slice surrogates, here evaluated on decoded points.
"""
import torch
import torch.nn.functional as F


def growth_monotonicity_loss(decoder, latent, coords, ambient_norm,
                             growth_frac=0.6, dt=0.02):
    """Penalize dT/dt < 0 within the growth phase, on sampled coords.

    coords: (B, Q, 3) with column 0 = normalized time.
    """
    t = coords[..., 0]
    mask = (t < growth_frac).float()                 # (B, Q)
    if mask.sum() == 0:
        return latent.new_zeros(())

    coords_next = coords.clone()
    coords_next[..., 0] = torch.clamp(coords[..., 0] + dt, max=1.0)

    T0 = decoder(latent, coords, ambient_norm)        # (B, Q)
    T1 = decoder(latent, coords_next, ambient_norm)   # (B, Q)

    decrease = F.relu(T0 - T1)                        # >0 where temperature drops
    return (decrease * mask).sum() / mask.sum().clamp(min=1.0)


def energy_hrr_loss(pred, hrr_norm):
    """Penalize negative correlation between mean decoded T and HRR across the batch.

    pred: (B, Q) decoded temperatures; hrr_norm: (B,) normalized HRR.
    """
    if pred.shape[0] < 3:
        return pred.new_zeros(())
    mean_t = pred.mean(dim=1)                          # (B,)
    mt = mean_t - mean_t.mean()
    hh = hrr_norm - hrr_norm.mean()
    denom = (mt.std() * hh.std()).clamp(min=1e-6)
    corr = (mt * hh).mean() / denom
    return F.relu(-corr)                               # 0 when positively correlated
