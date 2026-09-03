r"""Shared Streamlit input layer for the BS 8414 Part1 surrogate apps.

SHARED FILE — byte-identical in `bs8414_KAN_surrogate`, `bs8414_MLP_surrogate`,
`bs8414_fundiff_surrogate` and `bs8414_fundiff_kan_surrogate`. Never hand-edit
one copy: edit this file and re-copy it, exactly as for the `*_part1.py` data and
training modules.

Why it exists: the thermocouple apps and the slice apps must build the *same*
16-d parameter vector from the same build-up, because that is the only thing
that makes a TC prediction and a slice prediction comparable readings of one
case. Two independent copies of the material lookup and the normalisation call
would be free to drift, and the drift would be invisible — both apps would still
produce plausible curves.

Everything here reads `app_assets/part1_materials.json` (written by
`..\export_app_assets.py`) and the shared `data_loader_part1.normalize_material`.
The FDS corpus is never touched at runtime.
"""
import json
import os
import sys

import matplotlib
import numpy as np
import streamlit as st

# Launch-context guard: `streamlit run` puts this directory on sys.path, but
# other entry points (the headless AppTest harness, `python -m`) do not. Same
# idiom as `model_part1.py` / `dataset_part1.py`.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_part1 import (
    GEOMETRY_BITS, GEOMETRY_FLAGS, GEOMETRY_NAMES, MATERIAL_FEATURES,
    N_INPUT_PARAMS, N_TIMESTEPS, T_END,
)
from data_loader_part1 import normalize_material

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSET_PATH = os.path.join(PROJECT_DIR, "app_assets", "part1_materials.json")
SELECTION_PATH = os.path.join(PROJECT_DIR, "app_assets", "selected_model.json")

# The canonical reporting grid: 0 - 1800 s at 10 s, 181 samples.
TIME_S = np.linspace(0.0, T_END, N_TIMESTEPS)

GEOMETRY_HELP = {
    "noair": "Ventilated air cavity removed — the cladding sits directly on the "
             "insulation. These decks also have no cavity slice planes.",
    "nogap": "Open joints between cladding panels closed.",
    "nocb": "Cavity barriers removed.",
}


def apply_plot_style():
    """Journal-style matplotlib defaults, matching the paper figures."""
    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "font.family": "Arial", "font.size": 10, "axes.linewidth": 1.1,
        "axes.labelsize": 12, "axes.titlesize": 12, "axes.labelweight": "bold",
        "xtick.major.size": 5, "xtick.major.width": 1.1, "xtick.minor.size": 2.5,
        "xtick.direction": "in", "xtick.top": True,
        "ytick.major.size": 5, "ytick.major.width": 1.1, "ytick.minor.size": 2.5,
        "ytick.direction": "in", "ytick.right": True,
        "legend.frameon": True, "legend.edgecolor": "black",
        "legend.framealpha": 1.0, "legend.fontsize": 8, "figure.dpi": 130,
    })


@st.cache_data(show_spinner=False)
def load_assets():
    """The shipped material table, or None if it has not been exported yet."""
    if not os.path.isfile(ASSET_PATH):
        return None
    with open(ASSET_PATH) as f:
        return json.load(f)


def require_assets():
    """Load the table or stop the app with the command that generates it."""
    assets = load_assets()
    if assets is None:
        st.error(
            f"Material table not found at `{ASSET_PATH}`.\n\n"
            f"Generate it once (it reads the FDS corpus; the app never does):\n\n"
            f"```powershell\ncd {PROJECT_DIR}\n.\\venv\\Scripts\\activate\n"
            f"python ..\\export_app_assets.py\n```")
        st.stop()
    return assets


# ── deployment model selection ────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_selection():
    """`app_assets/selected_model.json`, or None if none has been chosen.

    Written by `..\\select_best_model.py`. It names the one run whose weights are
    un-ignored in `.gitignore`, so on a fresh clone it is the only run present —
    and on this workstation, where every run is on disk, it is the default.
    """
    if not os.path.isfile(SELECTION_PATH):
        return None
    with open(SELECTION_PATH) as f:
        return json.load(f)


def selection_index(names, selection):
    """Position of the deployment-selected run in `names`, else 0."""
    if selection and selection.get("selected") in names:
        return names.index(selection["selected"])
    return 0


def selection_label(name, selection):
    """Mark the deployment-selected run in the picker."""
    if selection and name == selection.get("selected"):
        return f"{name}  ★ selected"
    return name


def selection_caption(selection, chosen):
    """Sidebar note on what was selected, on what basis, and how firmly.

    The margin qualifier is not decoration: this corpus has ~0.05 retrain
    variance in test R2 and the project treats deltas inside +/-0.02 as
    inconclusive, so a picker that presented the top run as *the* best model
    would be overclaiming.
    """
    if not selection:
        return ("No deployment model selected for this project — run "
                "`python ..\\select_best_model.py`.")

    lines = [f"deployment model **{selection['selected']}**, chosen from "
             f"{selection['n_candidates']} runs on {selection['criterion']}"]

    if selection.get("runner_up"):
        margin = selection.get("margin")
        if selection.get("separated_beyond_noise"):
            lines.append(
                f"ahead of `{selection['runner_up']}` by {margin:.4f} — outside "
                f"the ±{selection['noise_band']} band the project treats as "
                f"inconclusive")
        else:
            lines.append(
                f"ahead of `{selection['runner_up']}` by only {margin:.4f}, "
                f"**inside** the ±{selection['noise_band']} band this project "
                f"treats as inconclusive — best available, not significantly "
                f"best")

    if selection.get("validation_agrees") is False:
        lines.append(
            f"note: ranking on the validation split alone would have chosen "
            f"`{selection['validation_only_winner']}` instead")
    if selection.get("validation_note"):
        lines.append(selection["validation_note"])

    if chosen != selection["selected"]:
        lines.append(f"⚠ you are viewing `{chosen}`, not the selected model; "
                     f"its weights may not be in version control")
    return "  \n".join(lines)


def default_material(assets, cladding, insulation):
    """The 13 raw material features for this build-up, in MATERIAL_FEATURES order.

    Exact, not interpolated: `export_app_assets.py` refuses to write the table
    unless a cladding id fixes its core + reaction block and an insulation id
    fixes its insulation block across the whole corpus.
    """
    clad, ins = assets["cladding"][cladding], assets["insulation"][insulation]
    merged = {**clad["core"], **ins["ins"], **clad["reac"]}
    missing = [f for f in MATERIAL_FEATURES if f not in merged]
    if missing:
        raise RuntimeError(f"asset table is missing {missing} — re-run "
                           f"export_app_assets.py")
    return [float(merged[f]) for f in MATERIAL_FEATURES]


def build_params(assets, cladding, insulation, geom_id, material_raw):
    """The 16-d Part1 vector: 3 categorical ids + 13 normalised material feats.

    Identical construction to `data_loader_part1.build_dataset`, which is what
    the models were trained on.
    """
    vec = [float(assets["cladding"][cladding]["id"]),
           float(assets["insulation"][insulation]["id"]),
           float(geom_id)] + normalize_material(material_raw)
    if len(vec) != N_INPUT_PARAMS:
        raise RuntimeError(f"built {len(vec)} params, expected {N_INPUT_PARAMS}")
    return np.asarray(vec, dtype=np.float32)


def build_up_selector(assets):
    """Render the build-up controls + the extrapolation notice.

    Returns (cladding, insulation, geom_id, in_corpus).
    """
    st.subheader("Build-up")
    col1, col2, col3 = st.columns([1.1, 1.0, 1.4])
    with col1:
        cladding = st.selectbox(
            "Cladding system", list(assets["cladding"]),
            help="Core material is read from SURF 'ACM_Core' in the FDS decks.")
        st.caption(f"core MATL `{assets['cladding'][cladding]['core_matl']}` "
                   f"· {assets['cladding'][cladding]['n_decks']} decks")
    with col2:
        insulation = st.selectbox(
            "Insulation", list(assets["insulation"]),
            help="Resolved from the MATL bound to SURF 'Insulation'.")
        st.caption(f"MATL `{assets['insulation'][insulation]['ins_matl']}` "
                   f"· {assets['insulation'][insulation]['n_decks']} decks")
    with col3:
        st.markdown("**Geometry modifiers**")
        geom_id = 0
        for gcol, flag in zip(st.columns(3), GEOMETRY_FLAGS):
            with gcol:
                if st.checkbox(flag, help=GEOMETRY_HELP[flag]):
                    geom_id |= GEOMETRY_BITS[flag]
        st.caption(f"geom_id **{geom_id}** — `{GEOMETRY_NAMES[geom_id]}`")

    observed = assets["observed"].get(f"{cladding}|{insulation}")
    in_corpus = observed is not None and geom_id in observed
    if observed is None:
        st.warning(
            f"**Extrapolation.** The corpus contains no `{cladding}` + "
            f"`{insulation}` build-up at all, so these two embeddings were never "
            f"trained together. Treat the prediction as indicative only.")
    elif not in_corpus:
        seen = ", ".join(f"`{GEOMETRY_NAMES[g]}`" for g in observed)
        st.warning(
            f"**Extrapolation.** `{cladding}` + `{insulation}` exists in the "
            f"corpus, but only as {seen}. Geometry is an 8-way embedding over "
            f"observed flag combinations — it cannot extrapolate to a "
            f"combination absent from training.")
    else:
        st.success(f"In corpus: `{cladding}` + `{insulation}` + "
                   f"`{GEOMETRY_NAMES[geom_id]}` was simulated in FDS.")
    return cladding, insulation, geom_id, in_corpus


def material_editor(assets, cladding, insulation):
    """Render the 13 material features, prefilled from the table. Returns raws."""
    defaults = default_material(assets, cladding, insulation)
    state_key = f"{cladding}|{insulation}"
    if st.session_state.get("_material_key") != state_key:
        st.session_state["_material_key"] = state_key
        st.session_state["_material"] = list(defaults)

    with st.expander("Material properties "
                     "(13 features, as parsed from the FDS decks)"):
        st.caption("Defaults are the exact values the model was trained on for "
                   "this build-up. Overriding them probes the surrogate off its "
                   "training manifold — useful for sensitivity work, not for "
                   "a design claim.")
        edited, cols = [], st.columns(3)
        for i, feature in enumerate(MATERIAL_FEATURES):
            with cols[i % 3]:
                edited.append(st.number_input(
                    feature, value=float(st.session_state["_material"][i]),
                    format="%.4f", key=f"mat_{i}_{state_key}"))
        st.session_state["_material"] = edited
        if any(abs(a - b) > 1e-9 for a, b in zip(edited, defaults)):
            st.info("Material features differ from this build-up's FDS values.")
    return list(st.session_state["_material"])


def input_vector_table(params_vec, cladding, insulation, geom_id, material_raw):
    """Rows describing exactly what was fed to the model, for the Data tab."""
    return {
        "component": ["cladding_id", "insulation_id", "geom_id"]
        + list(MATERIAL_FEATURES),
        "model input": [f"{v:.6g}" for v in params_vec],
        "raw / name": [cladding, insulation, GEOMETRY_NAMES[geom_id]]
        + [f"{v:.6g}" for v in material_raw],
    }


def asset_caption(assets):
    return (f"Material table: {assets['n_simulations']} simulations, "
            f"exported {assets['generated_utc']}")
