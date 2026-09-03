r"""BS 8414 Part1 slice-field surrogate — Streamlit prediction app (FunDiff).

    venv\Scripts\activate  &&  streamlit run app_fundiff_part1.py

Generates the 2D temperature field T(t, z, y) on the BS 8414 slice planes —
181 timesteps x 128 (vertical Z) x 64 (horizontal) per plane — for any
cladding x insulation x geometry build-up, from a trained two-stage FunDiff run.

SHARED FILE — byte-identical in `bs8414_fundiff_surrogate` and
`bs8414_fundiff_kan_surrogate`. It binds only to the uniform Part1 interface
(`Part1Surrogate`, `MODEL_NAME` from `model_part1.py`), which is the one file
that differs between them. Never hand-edit one copy: edit this file and re-copy
it.

Readout, identical to `evaluate.py`'s frozen contract: the FDS field is
(near-)deterministic given the deck, so the generative model is read out as a
**conditional mean** — sample N latents from fixed-seed noise, average them,
then decode. Changing the seed or the sample count changes the readout, so both
are shown and both are recorded in the export.

Runtime dependencies are `fae_best.pt` + `dit_best.pt` (the normalisation stats
ride inside them) plus `app_assets/part1_materials.json`. Neither the FDS corpus
nor the extracted `part1_slices/*.npz` is read.

Physical honesty the app enforces:
  * A `noair` build-up has no ventilated cavity, so `Wing_cavity` and
    `Main_cavity` DO NOT EXIST for it. FDS writes 3 planes, not 5. The app
    refuses to generate them rather than returning a plausible cold cavity.
  * Geometry is an 8-way embedding over observed flag combinations and cannot
    extrapolate to a combination absent from training.
"""
import glob
import io
import json
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

# Launch-context guard: `streamlit run` puts this directory on sys.path, but
# other entry points (the headless AppTest harness, `python -m`) do not. Same
# idiom as `model_part1.py` / `dataset_part1.py`.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as parent_config
from app_common_part1 import (
    TIME_S, apply_plot_style, asset_caption, build_params, build_up_selector,
    input_vector_table, load_selection, material_editor, require_assets,
    selection_caption, selection_index, selection_label,
)
from config import DECODE_CHUNK, EMBED_DIM, N_LATENT_TOKENS
from config_part1 import (
    DEVICE, FIELD_HEIGHT, FIELD_WIDTH, GEOMETRY_NAMES, GEOM_HAS_CAVITY,
    N_TIMESTEPS, SLICE_ID_MAP, SLICE_IDS, SLICE_REQUIRES_CAVITY, T_AMBIENT,
    T_END,
)
from data_processing.dataset import full_grid_coords
from model.dit import sample_latent
from model.fae import FunctionAutoencoder, count_parameters
from model_part1 import MODEL_NAME, Part1Surrogate

apply_plot_style()

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Physical extent of the sampled planes, for axis labels only. The BS 8414 rig
# is 9 m tall over the 128-cell vertical axis; the in-plane width differs per
# plane, so the horizontal axis is left in cells.
RIG_HEIGHT_M = 9.0


# ── checkpoints ───────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def discover_runs():
    """Two-stage runs in this project, best recorded generative R2 first.

    A run is usable only if its DiT was trained with the **Part1** conditioner:
    the 60-sim `models/` directory in this project holds a DiT built on the
    parent `ParamConditioner`, whose state_dict cannot load into `Part1DiT`.
    That is detected here by key inspection rather than left to fail on click.
    """
    runs = []
    for path in sorted(glob.glob(os.path.join(PROJECT_DIR, "models*"))):
        fae = os.path.join(path, "fae_best.pt")
        dit = os.path.join(path, "dit_best.pt")
        if not (os.path.isfile(fae) and os.path.isfile(dit)):
            continue
        state = torch.load(dit, map_location="cpu",
                           weights_only=False).get("dit_state", {})
        is_part1 = any(k.startswith("cond.conditioner.") for k in state)

        score = None
        metrics = os.path.join(path, "metrics.json")
        if os.path.isfile(metrics):
            try:
                with open(metrics) as f:
                    score = json.load(f).get("dit_gen", {}) \
                                        .get("global", {}).get("r2")
            except (ValueError, OSError):
                score = None
        runs.append({"name": os.path.basename(path), "path": path,
                     "is_part1": is_part1, "dit_gen_r2": score,
                     "mtime": os.path.getmtime(dit)})
    runs.sort(key=lambda r: (not r["is_part1"], r["dit_gen_r2"] is None,
                             -(r["dit_gen_r2"] or 0.0), -r["mtime"]))
    return runs


@st.cache_resource(show_spinner=False)
def load_stages(model_dir):
    """Frozen FAE + Part1 DiT, and the normalisation stats they were fitted on."""
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    fae_ckpt = torch.load(os.path.join(model_dir, "fae_best.pt"),
                          map_location=device, weights_only=False)
    fae = FunctionAutoencoder(parent_config).to(device)
    fae.load_state_dict(fae_ckpt["fae_state"])
    fae.eval()

    dit_ckpt = torch.load(os.path.join(model_dir, "dit_best.pt"),
                          map_location=device, weights_only=False)
    dit = Part1Surrogate(parent_config).to(device)
    dit.load_state_dict(dit_ckpt["dit_state"])
    dit.eval()

    # Both stages save the stats; the FAE's are the ones the decoder's ambient
    # floor was trained against, which is what `evaluate.py` reads.
    norm_stats = fae_ckpt.get("norm_stats") or dit_ckpt["norm_stats"]
    info = {"fae_params": count_parameters(fae), "dit_params": count_parameters(dit)}
    return fae, dit, norm_stats, info, device


@torch.no_grad()
def generate_plane(fae, dit, norm_stats, params_vec, slice_name, device,
                   n_steps, n_samples, seed, progress=None):
    """Conditional-mean field for one plane -> (181, 128, 64) in degC."""
    stats = norm_stats.get(slice_name, norm_stats["global"])
    mean, std = float(stats["mean"]), float(stats["std"])

    params = torch.as_tensor(params_vec, dtype=torch.float32,
                             device=device).unsqueeze(0)
    slice_ids = torch.tensor([SLICE_ID_MAP[slice_name]], dtype=torch.long,
                             device=device)
    ambient = torch.tensor([(T_AMBIENT - mean) / std], dtype=torch.float32,
                           device=device)

    generator = torch.Generator(device=device).manual_seed(int(seed))
    latents = [sample_latent(dit, params, slice_ids, N_LATENT_TOKENS, EMBED_DIM,
                             n_steps=n_steps, generator=generator)
               for _ in range(n_samples)]
    z_mean = torch.stack(latents).mean(0)

    grid = full_grid_coords(device=device)          # (T*H*W, 3), normalised
    out = []
    for start in range(0, grid.shape[0], DECODE_CHUNK):
        chunk = grid[start:start + DECODE_CHUNK].unsqueeze(0)
        out.append(fae.decode(z_mean, chunk, ambient).squeeze(0))
        if progress is not None:
            progress(min(1.0, (start + DECODE_CHUNK) / grid.shape[0]))
    field = torch.cat(out).reshape(N_TIMESTEPS, FIELD_HEIGHT, FIELD_WIDTH)
    return field.cpu().numpy().astype(np.float32) * std + mean


def available_planes(geom_id):
    """Planes that physically exist for this geometry — cavity ones need a cavity."""
    has_cavity = GEOM_HAS_CAVITY[geom_id]
    return [s for s in SLICE_IDS if has_cavity or not SLICE_REQUIRES_CAVITY[s]]


# ── figures ───────────────────────────────────────────────────────────────

def field_figure(field, slice_name, step, vmax):
    """Heatmap of one timestep, vertical axis up (axis0 = Z per the 2026-07 fix)."""
    fig, ax = plt.subplots(figsize=(3.4, 5.4))
    im = ax.imshow(field[step], origin="lower", aspect="auto", cmap="inferno",
                   vmin=T_AMBIENT, vmax=vmax,
                   extent=[0, FIELD_WIDTH, 0.0, RIG_HEIGHT_M])
    ax.set_title(f"{slice_name}\nt = {TIME_S[step]:.0f} s", fontsize=10)
    ax.set_xlabel("in-plane (cells)")
    ax.set_ylabel("Height (m)")
    fig.colorbar(im, ax=ax, label="T (°C)", fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def peak_curve_figure(fields):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for name, field in fields.items():
        axes[0].plot(TIME_S, field.max(axis=(1, 2)), lw=1.5, label=name)
        axes[1].plot(TIME_S, field.mean(axis=(1, 2)), lw=1.5, label=name)
    for ax, title, ylabel in (
            (axes[0], "Peak temperature in plane", "max T (°C)"),
            (axes[1], "Plane-mean temperature", "mean T (°C)")):
        ax.axhline(T_AMBIENT, color="0.5", ls=":", lw=1.0)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xlim(0, T_END)
        ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def height_profile_figure(fields):
    """Time-max temperature against height — the flame-spread readout."""
    heights = np.linspace(0.0, RIG_HEIGHT_M, FIELD_HEIGHT)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    for name, field in fields.items():
        ax.plot(field.max(axis=0).max(axis=1), heights, lw=1.6, label=name)
    ax.axvline(T_AMBIENT, color="0.5", ls=":", lw=1.0)
    ax.set_xlabel("Time-max temperature (°C)")
    ax.set_ylabel("Height (m)")
    ax.set_title("Vertical temperature envelope")
    ax.set_ylim(0, RIG_HEIGHT_M)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


# ── app ───────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title=f"BS 8414 Part1 — {MODEL_NAME}",
                       page_icon="\U0001f525", layout="wide")
    st.title("BS 8414-1 facade fire — Part1 slice-field surrogate")
    st.caption(f"{MODEL_NAME} — {N_TIMESTEPS} timesteps × {FIELD_HEIGHT} "
               f"(vertical) × {FIELD_WIDTH}, 0–{int(T_END)} s")

    assets = require_assets()
    runs = discover_runs()
    part1_runs = [r for r in runs if r["is_part1"]]
    if not part1_runs:
        skipped = ", ".join(f"`{r['name']}`" for r in runs) or "none"
        st.error(
            f"No Part1 two-stage run found in `{PROJECT_DIR}` (checked for "
            f"`fae_best.pt` + a `dit_best.pt` carrying the Part1 conditioner; "
            f"directories seen: {skipped}).\n\nTrain one with "
            f"`PART1_MODEL_DIR=models_part1_fundiff_r1 python train_fae.py` "
            f"then `train_dit.py`.")
        st.stop()

    # ── sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Model")
        selection = load_selection()
        labels = {r["name"]: (selection_label(r["name"], selection)
                              + (f"  — R² {r['dit_gen_r2']:.4f}"
                                 if r["dit_gen_r2"] is not None else ""))
                  for r in part1_runs}
        names = [r["name"] for r in part1_runs]
        choice = st.selectbox("Trained run", names,
                              index=selection_index(names, selection),
                              format_func=lambda n: labels[n])
        run = next(r for r in part1_runs if r["name"] == choice)
        st.caption(selection_caption(selection, choice))

        with st.spinner("Loading FAE + DiT…"):
            fae, dit, norm_stats, info, device = load_stages(run["path"])

        st.caption(
            f"device **{device.type}**"
            + (f" · {torch.cuda.get_device_name(0)}"
               if device.type == "cuda" else "")
            + f"  \nFAE {info['fae_params']:,} · DiT {info['dit_params']:,} "
            f"parameters")
        if run["dit_gen_r2"] is not None:
            st.caption(f"recorded test DiT(gen) R² **{run['dit_gen_r2']:.4f}** "
                       f"(`metrics.json`)")
        else:
            st.caption("no `metrics.json` in this run — score unknown; "
                       "run `evaluate.py`")
        if len(runs) > len(part1_runs):
            hidden = ", ".join(f"`{r['name']}`" for r in runs
                               if not r["is_part1"])
            st.caption(f"hidden (not Part1 conditioning): {hidden}")

        st.divider()
        st.header("Sampling")
        st.caption("Rectified-flow readout. The FDS field is deterministic given "
                   "the deck, so the model is read out as a conditional mean.")
        n_steps = st.slider("ODE steps", 10, 100, parent_config.N_SAMPLING_STEPS,
                            step=5,
                            help="Midpoint-Euler integration steps from noise "
                                 "to data. The frozen evaluation contract uses "
                                 f"{parent_config.N_SAMPLING_STEPS}.")
        n_samples = st.slider("Conditional samples", 1, 16,
                              parent_config.N_COND_SAMPLES,
                              help="Latents averaged before decoding. The frozen "
                                   "evaluation contract uses "
                                   f"{parent_config.N_COND_SAMPLES}.")
        seed = st.number_input("Seed", value=int(parent_config.INFERENCE_SEED),
                               step=1,
                               help="Fixed noise seed; the readout is only "
                                    "reproducible for a given seed.")
        st.divider()
        st.caption(asset_caption(assets))

    # ── inputs ────────────────────────────────────────────────────────────
    cladding, insulation, geom_id, _ = build_up_selector(assets)
    material_raw = material_editor(assets, cladding, insulation)
    params_vec = build_params(assets, cladding, insulation, geom_id, material_raw)

    planes = available_planes(geom_id)
    if len(planes) < len(SLICE_IDS):
        absent = ", ".join(f"`{s}`" for s in SLICE_IDS if s not in planes)
        st.info(
            f"`{GEOMETRY_NAMES[geom_id]}` removes the ventilated cavity, so "
            f"{absent} do not exist for this build-up — FDS writes 3 slice "
            f"planes, not 5. They are not offered: a generated cavity field "
            f"here would be a fiction, not a prediction.")

    chosen = st.multiselect("Slice planes to generate", planes,
                            default=planes[:1],
                            help="Each plane is an independent conditional "
                                 "generation; generating all of them takes "
                                 "proportionally longer.")
    if not chosen:
        st.stop()

    est = len(chosen) * (n_samples / max(parent_config.N_COND_SAMPLES, 1))
    if st.button(f"Generate {len(chosen)} plane"
                 f"{'s' if len(chosen) != 1 else ''}", type="primary"):
        fields, timings = {}, {}
        bar = st.progress(0.0, text="Sampling…")
        for i, name in enumerate(chosen):
            t0 = time.time()

            def report(frac, i=i, name=name):
                bar.progress((i + frac) / len(chosen),
                             text=f"Decoding {name} ({i + 1}/{len(chosen)})…")

            fields[name] = generate_plane(
                fae, dit, norm_stats, params_vec, name, device,
                n_steps=n_steps, n_samples=n_samples, seed=seed,
                progress=report)
            timings[name] = time.time() - t0
        bar.empty()
        st.session_state["_fields"] = fields
        st.session_state["_timings"] = timings
        st.session_state["_field_key"] = (
            cladding, insulation, geom_id, tuple(chosen), n_steps, n_samples,
            int(seed), run["name"])

    fields = st.session_state.get("_fields")
    if not fields:
        st.caption(f"Nothing generated yet — roughly {est:.0f}× the "
                   f"single-plane cost at the current settings.")
        st.stop()

    key = st.session_state.get("_field_key")
    if key and (key[0], key[1], key[2]) != (cladding, insulation, geom_id):
        st.warning(f"Showing fields generated for `{key[0]}` + `{key[1]}` + "
                   f"`{GEOMETRY_NAMES[key[2]]}`. Press **Generate** to update "
                   f"them for the current build-up.")

    vmax = max(float(f.max()) for f in fields.values())
    st.subheader("Generated fields")
    m1, m2, m3 = st.columns(3)
    hottest = max(fields, key=lambda n: fields[n].max())
    m1.metric("Peak temperature", f"{vmax:.0f} °C", help=f"in {hottest}")
    peak_step = int(fields[hottest].max(axis=(1, 2)).argmax())
    m2.metric("Time of peak", f"{TIME_S[peak_step]:.0f} s")
    m3.metric("Generation time",
              f"{sum(st.session_state['_timings'].values()):.1f} s",
              help=", ".join(f"{n} {t:.1f} s"
                             for n, t in st.session_state["_timings"].items()))

    tabs = st.tabs(["Field snapshot", "Time series", "Vertical envelope", "Data"])

    with tabs[0]:
        step = st.slider("Time (s)", 0, N_TIMESTEPS - 1, peak_step,
                         format="%d", key="_step",
                         help=f"Index on the {N_TIMESTEPS}-step / "
                              f"{int(T_END)} s grid.")
        st.caption(f"t = {TIME_S[step]:.0f} s · common colour scale "
                   f"{T_AMBIENT:.0f}–{vmax:.0f} °C across planes")
        for col, name in zip(st.columns(min(len(fields), 3)), fields):
            with col:
                fig = field_figure(fields[name], name, step, vmax)
                st.pyplot(fig)
                plt.close(fig)
        if len(fields) > 3:
            for col, name in zip(st.columns(len(fields) - 3),
                                 list(fields)[3:]):
                with col:
                    fig = field_figure(fields[name], name, step, vmax)
                    st.pyplot(fig)
                    plt.close(fig)

    with tabs[1]:
        fig = peak_curve_figure(fields)
        st.pyplot(fig)
        plt.close(fig)
        st.dataframe(pd.DataFrame([{
            "plane": name,
            "peak (°C)": round(float(f.max()), 1),
            "time of peak (s)": float(TIME_S[int(f.max(axis=(1, 2)).argmax())]),
            "final plane-mean (°C)": round(float(f[-1].mean()), 1),
            "height of peak (m)": round(
                float(np.unravel_index(f.argmax(), f.shape)[1])
                / (FIELD_HEIGHT - 1) * RIG_HEIGHT_M, 2),
        } for name, f in fields.items()]), width="stretch", hide_index=True)

    with tabs[2]:
        fig = height_profile_figure(fields)
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Time-max temperature at each of the 128 vertical cells, "
                   "maximised across the in-plane axis. Axis 0 of the stored "
                   "field is true vertical Z (the 2026-07 geometry fix).")

    with tabs[3]:
        stem = f"{cladding}_{insulation}_{GEOMETRY_NAMES[geom_id]}"
        buf = io_npz(fields, params_vec, cladding, insulation, geom_id,
                     run["name"], n_steps, n_samples, int(seed))
        st.download_button(
            "Download fields (.npz, float32 °C)", buf,
            file_name=f"{stem}_slices.npz", mime="application/octet-stream",
            help="One array per plane, shape (181, 128, 64), plus the input "
                 "vector and the sampling settings that produced it.")
        st.markdown("**Input vector (16-d, as fed to the model)**")
        st.dataframe(pd.DataFrame(input_vector_table(
            params_vec, cladding, insulation, geom_id, material_raw)),
            width="stretch", hide_index=True)
        st.markdown("**Per-plane normalisation (from the checkpoint)**")
        st.dataframe(pd.DataFrame([{
            "plane": name,
            "mean (°C)": round(float(norm_stats.get(
                name, norm_stats["global"])["mean"]), 2),
            "sd (°C)": round(float(norm_stats.get(
                name, norm_stats["global"])["std"]), 2),
        } for name in fields]), width="stretch", hide_index=True)


def io_npz(fields, params_vec, cladding, insulation, geom_id, run_name,
           n_steps, n_samples, seed):
    """Pack the generated planes and their provenance into an .npz buffer."""
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        params=params_vec,
        cladding=cladding, insulation=insulation,
        geometry=GEOMETRY_NAMES[geom_id], geom_id=geom_id,
        model=MODEL_NAME, run=run_name,
        n_sampling_steps=n_steps, n_cond_samples=n_samples, seed=seed,
        time_s=TIME_S,
        **{name: field for name, field in fields.items()})
    buf.seek(0)
    return buf.getvalue()


if __name__ == "__main__":
    main()
