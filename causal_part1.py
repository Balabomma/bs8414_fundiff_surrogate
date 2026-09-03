"""Causal (interventional) explainability for the Part1 surrogates.

Companion to `explain_part1.py`. The review that module follows — Cremades,
Hoyas & Vinuesa, Int. J. Heat and Fluid Flow 112 (2025) 109662 — closes §2.5
with the limitation that motivates this file:

    "SHAP values detect the correlation between the input and the output, so
     their causal connection may not be directly inferred from the SHAP values,
     with causality and information-theory algorithms being complementary to
     the SHAP values."

So this module supplies the complement. Where SHAP asks *how much of this
prediction is attributable to geometry*, this asks *what happens if the geometry
changes* — Pearl's do-operator, estimated on the surrogate by re-querying it
with one design variable overwritten and everything else held fixed.

WHY THIS CORPUS CAN DO SOMETHING MOST SURROGATE STUDIES CANNOT
--------------------------------------------------------------
Interventional attribution on a black box is usually unfalsifiable: you can
compute the model's response to do(X=x), but you have no ground truth to check
it against, because the real system was never run under both conditions.

Part1 is a designed factorial experiment — 8 geometry combinations crossed over
cladding and insulation systems — so for many cases BOTH arms of the
intervention were actually simulated in FDS. The true average causal effect is
therefore identified, not assumed, and the surrogate's causal response can be
*scored* against it. `ground_truth_ace` estimates the effect from
system-matched pairs; `model_ace` estimates the same quantity from the model;
`causal_fidelity` compares them.

That gives a model-quality axis that R2 cannot express. A surrogate can fit the
corpus well while responding to a design change in the wrong direction — and a
surrogate used to screen facade designs is used for exactly that, so the causal
response is arguably the property that matters most. Measured example: the best
physics-off KAN scores test TC R2 0.864 while predicting +5.3 degC for removing
cavity barriers where FDS gives +7 to +21; a physics-constrained run scores a
worse 0.805 and predicts +10.2 degC, i.e. it is the less accurate model and the
more causally faithful one.

CONFOUNDING
-----------
The estimand here is the *interventional* effect of a design flag, and the
design is randomised by construction (variants are generated from a reference
deck, not observed in the field), so there is no back-door path to block: the
geometry flags are set by the experimenter. The one real confounder is finite
sampling — a geometry cell that happens to contain hotter cladding systems — and
`ground_truth_ace` controls for it by differencing WITHIN a base system rather
than across the corpus, which is also why it reports the number of matched pairs
alongside every effect.

The LES noise floor is the other trap, and it is large: unsigned pairwise |dT|
between any two geometry variants sits at 38-54 degC regardless of whether a
systematic effect exists, because a point thermocouple in a turbulent fire is
not reproducible at that resolution. Every effect here is therefore a SIGNED
mean over matched pairs, reported with the fraction of pairs sharing its sign as
a consistency measure. An effect whose sign is inconsistent across pairs is
reported as such rather than averaged into a confident-looking number.
"""
import numpy as np
import torch

from config_part1 import (GEOMETRY_BITS, GEOMETRY_NAMES, COL_GEOM,
                          COL_CLADDING, COL_INSULATION)

# Interventions this corpus identifies: (label, bit, gate)
# `gate` restricts the estimand to the subpopulation where the intervention is
# physically meaningful. Removing cavity barriers is only defined as an
# intervention where a cavity exists; pooling the two subpopulations averages a
# real +20 degC effect with a null one and reports the mean of a bimodal
# quantity, which is how the chimney effect gets lost.
INTERVENTIONS = [
    ("nocb | cavity present", "nocb", "cavity"),
    ("nocb | no cavity", "nocb", "no_cavity"),
    ("noair (remove cavity)", "noair", None),
    ("nogap (close joints)", "nogap", None),
]


def _gate_mask(geom_ids, gate):
    has_cavity = (geom_ids & GEOMETRY_BITS["noair"]) == 0
    if gate == "cavity":
        return has_cavity
    if gate == "no_cavity":
        return ~has_cavity
    return np.ones_like(has_cavity, dtype=bool)


# ──────────────────────────────────────────────────────────────────────
# Ground truth: the effect FDS actually produced
# ──────────────────────────────────────────────────────────────────────

def ground_truth_ace(tc, mask, meta, bit_name, gate=None, sensors=slice(None)):
    """ACE of setting one geometry bit, from system-matched FDS pairs.

    Differences within a base system (same cladding + insulation), so cladding
    composition cannot confound the geometry contrast. Returns mean effect in
    degC, the sign-consistency fraction, and the number of matched pairs.
    """
    bit = GEOMETRY_BITS[bit_name]
    by_system = {}
    for i, m in enumerate(meta):
        by_system.setdefault(m["base"], {})[m["geom_id"]] = i

    deltas = []
    for pairs in by_system.values():
        for g_off, i in pairs.items():
            if g_off & bit:
                continue
            g_on = g_off | bit
            j = pairs.get(g_on)
            if j is None:
                continue
            if gate is not None and not _gate_mask(np.array([g_off]), gate)[0]:
                continue
            mm = (mask[i] > 0) & (mask[j] > 0)
            if mm.sum() == 0:
                continue
            deltas.append(float(tc[j][mm][:, sensors].mean()
                                - tc[i][mm][:, sensors].mean()))

    d = np.asarray(deltas)
    if d.size == 0:
        return {"ace_degC": float("nan"), "consistency": float("nan"), "n_pairs": 0}
    same_sign = (d > 0).mean() if d.mean() > 0 else (d < 0).mean()
    return {"ace_degC": float(d.mean()), "consistency": float(same_sign),
            "n_pairs": int(d.size)}


# ──────────────────────────────────────────────────────────────────────
# Model: the effect the surrogate believes in
# ──────────────────────────────────────────────────────────────────────

def _set_bit(params, bit, on):
    out = params.clone()
    g = out[:, COL_GEOM].long()
    out[:, COL_GEOM] = ((g | bit) if on else (g & ~bit)).to(out.dtype)
    return out


@torch.no_grad()
def model_ace(mean_fn, params, bit_name, gate=None):
    """ACE of the same intervention, estimated on the surrogate by do(bit).

    Every case is evaluated under BOTH arms, so this is a within-case paired
    contrast — the model's counterpart of the matched-pair design, with no
    sampling error from which cases happen to exist.
    """
    bit = GEOMETRY_BITS[bit_name]
    geom = params[:, COL_GEOM].long().numpy()
    keep = _gate_mask(geom, gate)
    if keep.sum() == 0:
        return {"ace_degC": float("nan"), "consistency": float("nan"), "n": 0}

    p = params[torch.as_tensor(keep)]
    d = (mean_fn(_set_bit(p, bit, True)) - mean_fn(_set_bit(p, bit, False)))
    d = d.detach().cpu().numpy()
    same_sign = (d > 0).mean() if d.mean() > 0 else (d < 0).mean()
    return {"ace_degC": float(d.mean()), "consistency": float(same_sign),
            "n": int(d.size)}


# ──────────────────────────────────────────────────────────────────────
# Scoring
# ──────────────────────────────────────────────────────────────────────

def causal_fidelity(mean_fn, params, tc, mask, meta, interventions=None):
    """Score the surrogate's causal response against the FDS-measured one.

    Per intervention: the true ACE, the model's ACE, whether they agree in sign,
    and the ratio of magnitudes. `sign_ok` is the property that matters for
    design screening — a surrogate that moves the wrong way when a barrier is
    removed is unusable for that purpose whatever its R2.
    """
    rows = []
    for label, bit_name, gate in (interventions or INTERVENTIONS):
        truth = ground_truth_ace(tc, mask, meta, bit_name, gate)
        pred = model_ace(mean_fn, params, bit_name, gate)
        t, p = truth["ace_degC"], pred["ace_degC"]
        sign_ok = bool(np.isfinite(t) and np.isfinite(p) and np.sign(t) == np.sign(p))
        ratio = float(p / t) if (np.isfinite(t) and abs(t) > 1e-6) else float("nan")
        rows.append({"intervention": label,
                     "truth_ace_degC": t, "truth_consistency": truth["consistency"],
                     "n_pairs": truth["n_pairs"],
                     "model_ace_degC": p, "model_consistency": pred["consistency"],
                     "sign_ok": sign_ok, "magnitude_ratio": ratio})
    return rows


def fidelity_score(rows, weight_by_consistency=True):
    """One number in [0, 1]: how faithfully the model reproduces the true ACEs.

    Per intervention the score is 1 - |log(ratio)| clipped to [0, 1] when the
    sign is right, and 0 when it is wrong — a model that halves or doubles the
    effect loses ~0.69, a model that reverses it scores nothing for that term.
    Interventions whose ground-truth sign is itself inconsistent across matched
    pairs carry proportionally less weight, because there the truth is weak
    rather than the model.
    """
    num = den = 0.0
    for r in rows:
        if not np.isfinite(r["truth_ace_degC"]):
            continue
        w = r["truth_consistency"] if weight_by_consistency else 1.0
        if not np.isfinite(w):
            w = 1.0
        s = 0.0
        if r["sign_ok"] and np.isfinite(r["magnitude_ratio"]) \
                and r["magnitude_ratio"] > 0:
            s = max(0.0, 1.0 - abs(np.log(r["magnitude_ratio"])))
        num += w * s
        den += w
    return float(num / den) if den else float("nan")


def format_report(rows, title=""):
    out = []
    if title:
        out.append(title)
    out.append(f"  {'intervention':<24}{'FDS ACE':>10}{'cons':>7}{'pairs':>7}"
               f"{'model ACE':>11}{'sign':>6}{'ratio':>8}")
    for r in rows:
        cons = f"{r['truth_consistency']*100:.0f}%" if np.isfinite(
            r["truth_consistency"]) else "-"
        ratio = f"{r['magnitude_ratio']:.2f}" if np.isfinite(
            r["magnitude_ratio"]) else "-"
        out.append(f"  {r['intervention']:<24}{r['truth_ace_degC']:>+10.2f}"
                   f"{cons:>7}{r['n_pairs']:>7}{r['model_ace_degC']:>+11.2f}"
                   f"{'ok' if r['sign_ok'] else 'WRONG':>6}{ratio:>8}")
    out.append(f"  causal fidelity score: {fidelity_score(rows):.3f}")
    return "\n".join(out)
