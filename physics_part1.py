"""Physics constraints for the Part1 geometry-variant corpus.

Every constraint here was **measured in the FDS ground truth before it was
imposed**. That is the same rule `evaluate_part1.physics_gates` follows, and it
is not ceremony: the first version of this module asserted that removing cavity
barriers is a no-op when there is no ventilated cavity, which is textbook facade
physics and which the corpus flatly refuses to confirm on a pointwise basis.
The measurement is what turned that into a constraint the data actually
supports. Diagnostics that produced the numbers below: `_diag_geom_physics.py`,
185 usable cases, system-matched pairs (same cladding + insulation, one geometry
bit toggled).

WHAT WAS MEASURED
-----------------
1. Ambient floor, T >= TMPA = 18 degC — holds, corpus minimum 17.88 degC,
   0/185 cases below. **Already enforced** as a hard clamp in the model
   (`ambient_scaled`), so this module does not re-impose it.

2. Energy-budget closure. Q_TOTAL is not an independent quantity; across all
   reported timesteps

       Q_TOTAL - (HRR + Q_RADI + Q_CONV + Q_COND) = 0.31 +/- 8.08 kW

   against a Q_TOTAL standard deviation of 17.93 kW. So the identity explains
   ~80 % of Q_TOTAL's variance, and the 8 kW slack is the Q_PRES / Q_PART /
   Q_DIFF terms that Part1 deliberately does not predict. This is why every
   architecture scores R2 0.09-0.39 on Q_TOTAL while scoring 0.94+ on the other
   four: Q_TOTAL is a small residual of large numbers, and nothing in the loss
   tells the heads they must balance.

   NOTE the constraint does **not** reconstruct Q_TOTAL from the other channels.
   HRR alone carries ~300 kW of model error, so a derived Q_TOTAL would be far
   worse than a predicted one. It penalises the *closure residual* instead,
   which couples the four large channels' errors without inheriting their
   magnitude.

3. Cavity-gated chimney effect — the geometry law this corpus exists to study.
   Signed, system-matched mean thermocouple shift from setting `nocb`
   (cavity barriers removed):

       cavity present   g4-g0   LV1 +20.75  LV2 +19.70   consistent 10/10
       cavity present   g6-g2   LV1  +7.12  LV2 +10.28   consistent  7/10
       NO cavity        g5-g1   LV1  -4.11  LV2  -6.95   consistent  3-4/11
       NO cavity        g7-g3   LV1  -3.04  LV2  -4.98   consistent  3-4/11

   Removing cavity barriers heats the facade **only when there is a cavity for
   them to bar**. Without one the effect collapses by a factor of ~4 and loses
   its sign. Removing the cavity itself cools, and does so almost without
   exception:

       g1-g0  LV1 -27.58 (8/10 cooler)    g5-g4  LV1 -34.36 (21/21 cooler)
       g3-g2  LV1 -25.48 (11/11 cooler)   g5-g4  LV2 -21.51 (19/21 cooler)

   An unsigned |dT| test does NOT show this: absolute pairwise differences sit
   at 38-54 degC for every geometry pair including the ones with no systematic
   effect, because LES scatter dominates pointwise. Only the signed,
   system-matched mean separates physics from turbulence. Any future constraint
   added here must be justified the same way.

WHY A COUNTERFACTUAL CONSTRAINT
-------------------------------
Geometry enters the model as an 8-way `nn.Embedding` over observed flag
combinations. That representation can fit any of the 8 ids it has seen and knows
nothing about how they relate — id 5 is not "id 1 with barriers removed" to it,
just a different atom. The project notes this as a known limitation: it "cannot
extrapolate to a combination absent from training".

The ordering constraint is applied by re-querying the model on the *same* case
with one geometry bit toggled and penalising predictions that violate the
measured ordering. No ground truth is consumed, so it applies to every training
case and to flag combinations that were never simulated — it teaches the
structure of the geometry axis rather than eight unrelated offsets.

WHAT THE TRAINED MODEL ACTUALLY BELIEVES
----------------------------------------
Measured on `models_part1_r4_seed48/part1_member0.pt` (the best KAN run,
test TC R2 0.864) over its own 142 training cases, by re-querying it with one
geometry bit toggled — i.e. asking the model the same counterfactual question
the constraint asks:

    +nocb, cavity present :  +2.51 degC   (FDS says +7 .. +21)
    +nocb, no cavity      :  -2.58 degC   (FDS says -3 .. -7)     correct
    +noair, cavity removed: -20.44 degC   (FDS says -11 .. -34)   correct

The model has learned that removing the cavity cools the facade, and that
barriers do nothing once the cavity is gone. It has **almost entirely missed the
chimney effect itself** — it predicts a quarter to a tenth of the true heating
from removing cavity barriers, and 57 % of its training cases violate even the
deliberately-slack 4 degC hinge. That is an independent confirmation of what the
evaluator already showed from the other direction: `nocb` is the worst geometry
in every per-geometry breakdown (MLP r4_s52 scores TC R2 0.657 on it against
0.908 on baseline, at n=5, the largest test cell).

Worth stating plainly, because it bounds what any of this can achieve: the
systematic nocb effect is ~+20 degC on a corpus whose mean thermocouple reading
is 243 degC and whose peak is 935 degC, while current pointwise model RMSE is
30-100 degC. Those are different quantities — a systematic mean shift is still
recoverable in aggregate under larger random error — but it does mean the
geometry signal cannot be read off a single predicted case, and it is why the
constraint is written on *case-mean* temperatures rather than pointwise. What it
can fix is a model that has the effect's sign or magnitude wrong on average;
what it cannot fix is per-case scatter.

CALIBRATION
-----------
Term magnitudes on that same trained checkpoint, against a base loss
(tc + 0.3*hrr) of 0.3766:

    energy closure    3182.83     ->  lambda 1.2e-05 for ~10 % of base
    geometry ordering   43.01     ->  lambda 8.8e-04 for ~10 % of base

The closure term is large because it is normalised by the truth's own 8.08 kW
residual spread, while the model's HRR error is ~400 kW — that ratio is the
honest statement of how far from balanced the heads currently are, so the scale
is kept and the lambda absorbs it. Both numbers are KAN-derived; a different
architecture has a different base loss, so re-run the calibration rather than
inheriting these blind.

STATUS: every lambda defaults to 0.0, i.e. OFF. Existing Part1 checkpoints were
trained without these terms and stay comparable until a run opts in via the
environment variables below.

    PART1_LAMBDA_CLOSURE=1.2e-5   energy-budget closure
    PART1_LAMBDA_GEOM=8.8e-4      cavity-gated chimney ordering
"""
import os

import torch

from config_part1 import (HRR_CHANNELS, GEOMETRY_BITS, COL_GEOM,
                          SLICE_IDS, SLICE_REQUIRES_CAVITY)

# ── weights (0.0 = off; see module docstring) ─────────────────────────────
LAMBDA_CLOSURE = float(os.environ.get("PART1_LAMBDA_CLOSURE", "0.0"))
LAMBDA_GEOM = float(os.environ.get("PART1_LAMBDA_GEOM", "0.0"))

# ── measured constants ────────────────────────────────────────────────────
# Std of the closure residual in the ground truth, kW. The closure term is
# divided by this so a model that is as balanced as FDS itself scores ~1.0
# and the lambda means the same thing regardless of channel units.
CLOSURE_RESIDUAL_KW = 8.08

# Hinge margins, degC. Deliberately set well INSIDE the measured effects: the
# weakest confirmed cavity-present nocb shift is +7.1 degC, so a 4 degC margin
# penalises only unambiguous sign errors and never argues with a case whose true
# effect is merely smaller than average. Same logic for the others.
NOCB_CAVITY_MARGIN_C = 4.0     # measured +7.1 .. +20.8
NOCB_NOCAVITY_TOL_C = 10.0     # measured |-3.0| .. |-7.0|
NOAIR_COOLING_MARGIN_C = 8.0   # measured -10.8 .. -34.4

NOAIR_BIT = GEOMETRY_BITS["noair"]
NOCB_BIT = GEOMETRY_BITS["nocb"]

_iH, _iR, _iC, _iK, _iT = (HRR_CHANNELS.index(c) for c in
                           ("HRR", "Q_RADI", "Q_CONV", "Q_COND", "Q_TOTAL"))

# Planes that live inside the ventilated cavity and therefore do not exist in a
# `noair` deck. Derived from config rather than hard-coded so a change to the
# slice set cannot silently desynchronise this.
CAVITY_PLANE_IDS = tuple(i for i, name in enumerate(SLICE_IDS)
                         if SLICE_REQUIRES_CAVITY[name])


def planes_without_cavity(slice_ids):
    """(B,) bool: True where the plane exists whether or not there is a cavity."""
    ok = torch.ones_like(slice_ids, dtype=torch.bool)
    for i in CAVITY_PLANE_IDS:
        ok = ok & (slice_ids != i)
    return ok


def closure_residual(hrr_physical):
    """Q_TOTAL - (HRR + Q_RADI + Q_CONV + Q_COND) in kW, over (..., channel)."""
    return hrr_physical[..., _iT] - (hrr_physical[..., _iH]
                                     + hrr_physical[..., _iR]
                                     + hrr_physical[..., _iC]
                                     + hrr_physical[..., _iK])


def energy_closure_loss(hrr_pred, hrr_true, mask, hrr_mean, hrr_scale):
    """Penalise the model's energy-budget residual for missing the truth's.

    hrr_pred / hrr_true are standardised (B, T, C); hrr_mean / hrr_scale are the
    fitted per-channel statistics, so the residual is formed in physical kW where
    the identity actually lives. Matching the truth's residual rather than
    driving it to zero is the point — the truth's is 0.31 +/- 8.08 kW, not zero,
    because Part1 does not predict the Q_PRES / Q_PART terms.
    """
    pred = hrr_pred * hrr_scale + hrr_mean
    true = hrr_true * hrr_scale + hrr_mean
    err = (closure_residual(pred) - closure_residual(true)) / CLOSURE_RESIDUAL_KW
    err = (err ** 2) * mask
    return err.sum() / mask.sum().clamp(min=1.0)


def _toggled(params, bit, on):
    """Copy of `params` with one geometry bit forced on or off."""
    out = params.clone()
    geom = out[:, COL_GEOM].long()
    geom = (geom | bit) if on else (geom & ~bit)
    out[:, COL_GEOM] = geom.to(out.dtype)
    return out


def geometry_ordering_loss(mean_fn, params, max_samples=8, noair_eligible=None):
    """Cavity-gated chimney ordering, imposed by counterfactual re-query.

    `mean_fn(params) -> (B,)` must return the mean predicted temperature in degC
    for each case; the caller supplies it because the thermocouple models return
    (B, T, C) and the slice models return (B, T, 1, H, W), but the physics is
    identical either way.

    Three hinges, all one-sided, all silent unless the measured ordering is
    actually violated:

      cavity present -> removing barriers must not cool     (measured +7 .. +21)
      no cavity      -> removing barriers must do little    (measured |-3 .. -7|)
      cavity removed -> must cool                           (measured -11 .. -34)

    `noair_eligible` is a (B,) bool mask limiting which samples take part in the
    cavity-removal hinge. The slice pipeline needs it: two of the five planes are
    *inside* the cavity and simply do not exist in a `noair` deck, which is why
    the loader masks them rather than imputing zeros. Toggling `noair` on while
    asking about a cavity plane requests a field that has no physical referent,
    so those samples sit the third hinge out. The barrier hinges are unaffected —
    removing cavity barriers never changes which planes exist.
    """
    if params.shape[0] > max_samples:
        params = params[:max_samples]
        if noair_eligible is not None:
            noair_eligible = noair_eligible[:max_samples]

    geom = params[:, COL_GEOM].long()
    has_cavity = (geom & NOAIR_BIT) == 0

    # --- barriers removed vs present, holding everything else fixed ---
    t_barriers = mean_fn(_toggled(params, NOCB_BIT, on=False))
    t_no_barriers = mean_fn(_toggled(params, NOCB_BIT, on=True))
    d_nocb = t_no_barriers - t_barriers

    with_cav = torch.relu(NOCB_CAVITY_MARGIN_C - d_nocb)[has_cavity]
    without_cav = torch.relu(d_nocb.abs() - NOCB_NOCAVITY_TOL_C)[~has_cavity]

    # --- cavity removed vs present ---
    if noair_eligible is not None and not bool(noair_eligible.any()):
        cooling = params.new_zeros((0,))
    else:
        t_cavity = mean_fn(_toggled(params, NOAIR_BIT, on=False))
        t_no_cavity = mean_fn(_toggled(params, NOAIR_BIT, on=True))
        d_noair = t_no_cavity - t_cavity
        if noair_eligible is not None:
            d_noair = d_noair[noair_eligible]
        cooling = torch.relu(d_noair + NOAIR_COOLING_MARGIN_C)

    terms = [t for t in (with_cav, without_cav, cooling) if t.numel() > 0]
    if not terms:
        return torch.zeros((), device=params.device)
    return torch.cat([t.reshape(-1) for t in terms]).pow(2).mean()


def tc_mean_fn(model, time_array, tc_mean, tc_scale, mask=None):
    """mean_fn for the thermocouple models: (B, T, C) standardised -> (B,) degC."""
    def fn(params):
        tc = model(params, time_array)[0] * tc_scale + tc_mean
        if mask is None:
            return tc.mean(dim=(1, 2))
        m = mask.unsqueeze(-1)
        return (tc * m).sum(dim=(1, 2)) / m.sum(dim=(1, 2)).clamp(min=1.0)
    return fn


def slice_mean_fn(model, slice_ids, time_window, field_mean, field_scale):
    """mean_fn for the slice models: (B, T, 1, H, W) standardised -> (B,) degC."""
    def fn(params):
        field = model(params, slice_ids, time_window) * field_scale + field_mean
        return field.mean(dim=tuple(range(1, field.dim())))
    return fn


def describe():
    """One-line summary for the training log."""
    if not (LAMBDA_CLOSURE or LAMBDA_GEOM):
        return "physics constraints: off"
    bits = []
    if LAMBDA_CLOSURE:
        bits.append(f"{LAMBDA_CLOSURE} * energy-closure")
    if LAMBDA_GEOM:
        bits.append(f"{LAMBDA_GEOM} * geometry-ordering")
    return "physics constraints: " + " + ".join(bits)
