"""Additive-feature-attribution (SHAP) explainability for the Part1 surrogates.

Method and terminology follow Cremades, Hoyas & Vinuesa, "Additive-feature-
attribution methods: A review on explainable artificial intelligence for fluid
dynamics and heat transfer", Int. J. Heat and Fluid Flow 112 (2025) 109662.
The review's three axioms — local accuracy, missingness, consistency — are what
make Shapley values the unique additive attribution, and local accuracy is
checked numerically here rather than assumed (`self_test`).

WHY NOT gradient SHAP, WHICH THE REVIEW RECOMMENDS FOR DEEP MODELS
------------------------------------------------------------------
Section 2.6 recommends gradient SHAP when the architecture is accessible and the
model is a deep network. That advice does not survive contact with this
architecture, and the failure is silent rather than noisy. All three design
variables enter the model as integer ids consumed by `nn.Embedding` through
`.long()`:

    d(output)/d(cladding_id)   = 0.000e+00
    d(output)/d(insulation_id) = 0.000e+00
    d(output)/d(geom_id)       = 0.000e+00
    d(output)/d(material_*)    ~ 3e-03 .. 1e-02   (measured, untrained model)

`.long()` is piecewise constant, so every gradient-based attribution — gradient
SHAP, deep SHAP, integrated gradients, expected gradients — returns *exactly*
zero for the ids. A gradient-SHAP study of this model would therefore conclude
that geometry has no influence on the prediction, on a corpus that exists
specifically to measure the influence of geometry. It would look like a finding.

So the ids are attributed by EXACT Shapley values instead, which is affordable
here for a reason particular to this problem: the review's cost objection
(2^N coalitions, N ~ 1e9 for turbulence fields) does not apply when the
attribution is taken over three physical SYSTEMS rather than sixteen columns.
Three players is 8 coalitions — exact Eq. (6), no sampling, no approximation.

GROUPING IS NOT A CONVENIENCE, IT IS THE DEPENDENCE FIX
-------------------------------------------------------
Section 2.5 warns that SHAP assumes feature independence and loses accuracy when
inputs are strongly correlated. In this corpus the correlation is not merely
strong, it is deterministic: `cladding_id` *is* a lookup key for the six core
material features and the two reaction features, and `insulation_id` for the
five insulation features — `data_loader_part1.material_vector` parses them from
the deck the id names. Attributing to `core_conductivity` while holding
`cladding_id` fixed asks what the model would predict for a cladding system that
cannot exist.

The three groups below therefore move as physical units, which makes the
coalitions physically realisable inputs rather than chimeras:

    cladding_system   = cladding_id + core{cp,k,rho,HoR,T_ref,reactive}
                                    + reac{HoC,soot}
    insulation_system = insulation_id + ins{cp,k,rho,HoR,T_ref}
    geometry          = geom_id

`check_group_determinism()` verifies the premise against the corpus rather than
asserting it, and it currently finds **one exception in 185 cases**, which is a
corpus defect rather than a grouping error:

    BS8414_DCLG_Test7 declares insulation system PF, like 33 other decks, but
    its `Phenolic_Foam` MATL is inert — HEAT_OF_REACTION absent (0 kJ/kg) with
    REFERENCE_TEMPERATURE = 429 degC — where every other PF deck decomposes with
    HEAT_OF_REACTION = 400 kJ/kg and no explicit reference temperature
    (cp 1.2 vs 1.5, k 0.020 vs 0.022, rho 32 vs 39).

That is the same defect class the loader already excludes for
`BS8414_DCLG_Test1_adv_debug` ("a different system under the same name"), in the
same legacy DCLG family — but this one is NOT excluded and is in the training
split. It is left in deliberately: removing a case changes the corpus and every
number computed from it, which is the user's call, not this module's. Re-run the
check when the corpus grows.

REFERENCE VALUES
----------------
Section 2.5 stresses that the reference (baseline) conditions the whole result.
Zero or mean baselines are rejected here for the same reason as above: a mean
material vector belongs to no real cladding. Baselines are drawn from actual
TRAINING rows, so `f(S)` is always the model evaluated on a physically
constructible case, and the reported attribution is an expectation over that
reference distribution.

WHAT THIS DOES AND DOES NOT TELL YOU
------------------------------------
Section 2.5, last paragraph: SHAP values are correlational, and causal
connections cannot be read off them directly — the review names causality and
information-theory methods as complementary. This project already has the
complementary interventional measure: `physics_part1.geometry_ordering_loss`
toggles a single geometry bit and re-queries the model, which is an intervention
in the causal sense, and its ground-truth counterpart is measured in the FDS
corpus. `compare_with_intervention()` puts the two side by side, and they answer
different questions — SHAP says how much of THIS prediction is attributable to
geometry, the toggle says what happens IF geometry changes.
"""
import numpy as np
import torch

from config_part1 import (COL_CLADDING, COL_INSULATION, COL_GEOM,
                          MATERIAL_FEATURES, N_INPUT_PARAMS, GEOMETRY_NAMES)

# Column layout of the 16-d input, from data_loader_part1.material_vector:
#   0 cladding_id | 1 insulation_id | 2 geom_id
#   3-8   core {cp, k, rho, hor, tref, reactive}
#   9-13  ins  {cp, k, rho, hor, tref}
#   14-15 reac {hoc, soot}
CORE_COLS = list(range(3, 9))
INS_COLS = list(range(9, 14))
REAC_COLS = list(range(14, 16))

GROUPS = {
    "cladding_system": [COL_CLADDING] + CORE_COLS + REAC_COLS,
    "insulation_system": [COL_INSULATION] + INS_COLS,
    "geometry": [COL_GEOM],
}

CATEGORICAL_COLS = (COL_CLADDING, COL_INSULATION, COL_GEOM)
CONTINUOUS_COLS = [i for i in range(N_INPUT_PARAMS) if i not in CATEGORICAL_COLS]


# ──────────────────────────────────────────────────────────────────────
# Target reductions: a model output (B, T, C) -> one scalar per case
# ──────────────────────────────────────────────────────────────────────

def make_scalar_fn(model, time_array, tc_mean, tc_scale, reduction="mean",
                   growth_end_step=72):
    """Wrap a thermocouple model as (B, 16) -> (B,) degC.

    reduction: mean | lv1 | lv2 | peak | growth
    SHAP explains one scalar at a time; which scalar is a modelling choice, so
    it is explicit rather than hidden.
    """
    def fn(params):
        tc = model(params, time_array)[0] * tc_scale + tc_mean
        if reduction == "mean":
            return tc.mean(dim=(1, 2))
        if reduction == "lv1":
            return tc[:, :, :8].mean(dim=(1, 2))
        if reduction == "lv2":
            return tc[:, :, 8:].mean(dim=(1, 2))
        if reduction == "peak":
            return tc.amax(dim=(1, 2))
        if reduction == "growth":
            return tc[:, :growth_end_step + 1].mean(dim=(1, 2))
        raise ValueError(f"unknown reduction {reduction!r}")
    return fn


def make_slice_scalar_fn(model, slice_ids, time_window, field_mean, field_scale):
    """Wrap a slice-field model as (B, 16) -> (B,) degC (mean over the field)."""
    def fn(params):
        field = model(params, slice_ids, time_window) * field_scale + field_mean
        return field.mean(dim=tuple(range(1, field.dim())))
    return fn


# ──────────────────────────────────────────────────────────────────────
# Exact group Shapley values
# ──────────────────────────────────────────────────────────────────────

def _coalition_input(x, baseline, present_cols):
    """x where `present_cols` are kept, baseline elsewhere."""
    out = baseline.clone()
    out[:, present_cols] = x[:, present_cols]
    return out


@torch.no_grad()
def group_shapley(scalar_fn, x, baselines, groups=None, batch_size=256):
    """Exact Shapley values over the feature GROUPS. Eq. (6) of the review.

    x          (N, 16) cases to explain
    baselines  (M, 16) reference cases, drawn from the training split
    returns    dict name -> (N,) attribution in degC, averaged over baselines,
               plus "_base" (N,) mean f(baseline) and "_pred" (N,) f(x)

    With G groups this evaluates 2^G coalitions per baseline. G = 3 here, so it
    is 8 exact evaluations — the review's exponential-cost objection applies to
    per-column attribution, not to a three-player game.
    """
    groups = groups or GROUPS
    names = list(groups)
    G = len(names)
    N = x.shape[0]

    from itertools import combinations
    from math import factorial

    subsets = []
    for r in range(G + 1):
        subsets.extend(combinations(range(G), r))
    idx_of = {s: i for i, s in enumerate(subsets)}

    phi = {n: torch.zeros(N, dtype=torch.float64) for n in names}
    base_acc = torch.zeros(N, dtype=torch.float64)
    pred_acc = torch.zeros(N, dtype=torch.float64)

    for m in range(baselines.shape[0]):
        b = baselines[m:m + 1].expand(N, -1).contiguous()

        # value of every coalition, once
        vals = []
        for s in subsets:
            cols = [c for gi in s for c in groups[names[gi]]]
            inp = _coalition_input(x, b, cols) if cols else b
            outs = [scalar_fn(inp[i:i + batch_size])
                    for i in range(0, N, batch_size)]
            vals.append(torch.cat(outs).double().cpu())

        base_acc += vals[idx_of[()]]
        pred_acc += vals[idx_of[tuple(range(G))]]

        for gi, name in enumerate(names):
            for s in subsets:
                if gi in s:
                    continue
                w = factorial(len(s)) * factorial(G - len(s) - 1) / factorial(G)
                with_g = tuple(sorted(s + (gi,)))
                phi[name] += w * (vals[idx_of[with_g]] - vals[idx_of[s]])

    M = baselines.shape[0]
    out = {n: (phi[n] / M).numpy() for n in names}
    out["_base"] = (base_acc / M).numpy()
    out["_pred"] = (pred_acc / M).numpy()
    return out


def local_accuracy_error(attr):
    """Axiom 1: sum of attributions + E[f(baseline)] must equal f(x).

    Returned in degC. For exact Shapley this is zero to floating point, so a
    non-trivial value means the implementation is wrong, not that the model is.
    """
    names = [k for k in attr if not k.startswith("_")]
    total = sum(attr[n] for n in names) + attr["_base"]
    return np.abs(total - attr["_pred"])


# ──────────────────────────────────────────────────────────────────────
# Expected gradients — CONTINUOUS material features only
# ──────────────────────────────────────────────────────────────────────

def expected_gradients(scalar_fn, x, baselines, n_samples=64, seed=0):
    """Expected gradients (Erion et al. 2021), the review's gradient SHAP.

    Valid ONLY for the continuous columns: the id columns are consumed through
    `.long()` and their gradient is identically zero (see module docstring), so
    this returns them as NaN rather than as a confident zero that would read as
    "no influence".
    """
    g = torch.Generator().manual_seed(seed)
    N = x.shape[0]
    phi = torch.zeros(N, N_INPUT_PARAMS, dtype=torch.float64)

    for _ in range(n_samples):
        j = torch.randint(baselines.shape[0], (N,), generator=g)
        b = baselines[j]
        alpha = torch.rand(N, 1, generator=g)
        z = (b + alpha * (x - b)).detach().requires_grad_(True)
        out = scalar_fn(z).sum()
        grad, = torch.autograd.grad(out, z)
        phi += ((x - b) * grad).double().detach().cpu()

    phi /= n_samples
    phi[:, list(CATEGORICAL_COLS)] = float("nan")
    return phi.numpy()


# ──────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────

def global_importance(attr):
    """Local -> global: mean |phi| per group, and share of the explained total.

    Section 2.5 notes that global insight from local explanations needs care;
    mean absolute attribution is the standard aggregation and is reported with
    the signed mean beside it, because a group that pushes some cases up and
    others down has a large |phi| and a near-zero mean.
    """
    names = [k for k in attr if not k.startswith("_")]
    rows = []
    tot = sum(np.abs(attr[n]).mean() for n in names) or 1.0
    for n in names:
        rows.append({"group": n,
                     "mean_abs_degC": float(np.abs(attr[n]).mean()),
                     "mean_signed_degC": float(attr[n].mean()),
                     "share": float(np.abs(attr[n]).mean() / tot)})
    return sorted(rows, key=lambda r: -r["mean_abs_degC"])


def by_geometry(attr, geom_ids):
    """Geometry attribution split by which geometry the case actually is."""
    out = {}
    for g in sorted(set(int(v) for v in geom_ids)):
        sel = np.asarray(geom_ids) == g
        out[GEOMETRY_NAMES[g]] = {
            "n": int(sel.sum()),
            "geometry_phi_degC": float(attr["geometry"][sel].mean()),
        }
    return out


def compare_with_intervention(attr, geom_ids, intervention_deltas):
    """Put the correlational and interventional views side by side.

    `intervention_deltas` is a dict geometry-name -> degC shift measured by
    toggling the bit (physics_part1), i.e. a causal quantity. SHAP answers a
    different question and the two are not expected to be equal; a sign
    disagreement is the interesting case.
    """
    shap_side = by_geometry(attr, geom_ids)
    rows = []
    for name, d in shap_side.items():
        rows.append({"geometry": name, "n": d["n"],
                     "shap_phi_degC": d["geometry_phi_degC"],
                     "intervention_degC": intervention_deltas.get(name)})
    return rows


def check_group_determinism(params, meta=None, verbose=True):
    """Verify the premise the grouping rests on: an id fixes its material block.

    Returns a list of violations. An empty list means every coalition this
    module builds is a physically realisable case; a non-empty one means two
    decks share an id while describing different materials, and the grouped
    attribution for that id is averaging over two systems.
    """
    p = np.asarray(params)
    violations = []
    for idcol, cols, label in ((COL_CLADDING, CORE_COLS + REAC_COLS, "cladding"),
                               (COL_INSULATION, INS_COLS, "insulation")):
        for v in sorted(set(p[:, idcol].tolist())):
            sel = p[:, idcol] == v
            block = p[sel][:, cols]
            uniq, counts = np.unique(block, axis=0, return_counts=True)
            if len(uniq) > 1:
                odd = uniq[np.argmin(counts)]
                chids = ([m["chid"] for m, s in zip(meta, sel)
                          if s and (p[list(meta).index(m)][cols] == odd).all()]
                         if meta is not None else [])
                violations.append({"axis": label, "id": int(v),
                                   "n_distinct": int(len(uniq)),
                                   "minority_n": int(counts.min()),
                                   "chids": chids})
                if verbose:
                    print(f"  {label} id {int(v)}: {len(uniq)} distinct material "
                          f"vectors (minority n={counts.min()}) {chids}")
    if verbose and not violations:
        print("  every id maps to exactly one material vector")
    return violations


def self_test(scalar_fn, x, baselines, verbose=True):
    """Check the axiom the method rests on before trusting any number from it."""
    attr = group_shapley(scalar_fn, x, baselines)
    err = local_accuracy_error(attr)
    if verbose:
        print(f"  local accuracy |sum(phi) + E[f(base)] - f(x)|: "
              f"max {err.max():.3e} degC, mean {err.mean():.3e} degC")
        print(f"  -> {'PASS' if err.max() < 1e-3 else 'FAIL'}")
    return attr, err
