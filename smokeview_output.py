"""Smokeview-style rendering of FunDiff predictions.

Reuses the project's own Smokeview-equivalent renderer (the sibling `pysmokview`
package): predicted and ground-truth temperature fields saved by evaluate.py are
wrapped in a `SliceData` and rendered through `SliceAnimator` — the same dark
fire-engineering theme, TEMPERATURE colormap, and colorbar Smokeview produces.

For each saved (sim, slice) it writes:
    <chid>__<slice>_pred.gif   — animated predicted field
    <chid>__<slice>_true.gif   — animated ground-truth field
    <chid>__<slice>_cmp.png    — pred / truth / error still at mid-time

Run:  python smokeview_output.py         (after evaluate.py)
      python smokeview_output.py --mp4   (MP4 instead of GIF; needs ffmpeg)
"""
import os
import sys
import glob
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (OUTPUT_DIR, FIELD_HEIGHT, FIELD_WIDTH, N_TIMESTEPS, DT_SLCF,
                    T_AMBIENT, SLICE_GEOMETRY, STORED_AXIS0_IS_HORIZONTAL)

# Make the sibling pysmokview package importable (its deps: numpy/matplotlib/tqdm).
PYSMOKVIEW_ROOT = r"D:\VS_projects\pysmokview"
if os.path.isdir(PYSMOKVIEW_ROOT):
    sys.path.insert(0, PYSMOKVIEW_ROOT)

try:
    from pysmokview.core.slice_reader import SliceData
    from pysmokview.viz.slice_animator import SliceAnimator
    from pysmokview.config import Config
    HAVE_PYSMOKVIEW = True
except Exception as e:  # pragma: no cover
    HAVE_PYSMOKVIEW = False
    _IMPORT_ERR = e

def _to_physical(field, slice_name):
    """Reorient a stored (T, 128, 64) field to physical (T, Z_vert, horiz) in metres.

    Orients the stored field to physical (T, Z_vertical, horizontal) so Z is the
    first spatial axis (how a facade slice is viewed).  Whether a transpose is
    needed depends on the stored layout (config.STORED_AXIS0_IS_HORIZONTAL):
    post-2026-07-22 data already has axis0=Z, so no transpose.
    Returns (data (T, nz, nh), z_coords (nz,), h_coords (nh,), geom).
    """
    geom = SLICE_GEOMETRY[slice_name]
    if STORED_AXIS0_IS_HORIZONTAL:
        data = np.transpose(field, (0, 2, 1))             # (T, Z, horiz)
    else:
        data = np.asarray(field)                          # already (T, Z, horiz)
    nz, nh = data.shape[1], data.shape[2]
    z = np.linspace(geom["vert_range"][0], geom["vert_range"][1], nz)
    h = np.linspace(geom["horiz_range"][0], geom["horiz_range"][1], nh)
    return data.astype(np.float32), z, h, geom


def _slice_data(field, slice_name):
    """field: stored (T, 128, 64) physical degC -> SliceData in true metres."""
    data, z, h, geom = _to_physical(field, slice_name)
    times = np.arange(N_TIMESTEPS, dtype=float) * DT_SLCF
    return SliceData(
        data=data, times=times, x_coords=h, y_coords=z,   # x=horizontal(m), y=Z(m)
        quantity="TEMPERATURE", units="°C", orientation=geom["plane"],
        position=geom["pos"], meshes_used=[1], is_stitched=False,
    )


def _cmp_png(pred, true, slice_name, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pysmokview.viz.colormaps import get_colormap_config

    cm = get_colormap_config("TEMPERATURE")
    ti = N_TIMESTEPS // 2
    pdat, z, h, geom = _to_physical(pred, slice_name)
    tdat, _, _, _ = _to_physical(true, slice_name)
    # extent [horiz_min, horiz_max, z_min, z_max] so ticks read true metres
    extent = [h[0], h[-1], z[0], z[-1]]
    hlabel = f"{geom['horiz_axis'].upper()} (m)"
    vmax = max(np.percentile(true, 99), np.percentile(pred, 99), T_AMBIENT + 1)
    fig, ax = plt.subplots(1, 3, figsize=(13, 7))
    for a, (fld, ttl) in zip(ax, [(tdat[ti], "Ground truth (FDS)"),
                                  (pdat[ti], "FunDiff prediction")]):
        im = a.imshow(fld, origin="lower", cmap=cm["cmap"], vmin=T_AMBIENT, vmax=vmax,
                      extent=extent, aspect="equal")
        a.set_title(f"{ttl}\n{slice_name}  ({geom['plane']} @ {geom['normal']}={geom['pos']} m)")
        a.set_xlabel(hlabel); a.set_ylabel("Z (m)")
        plt.colorbar(im, ax=a, shrink=0.85, label="°C")
    err = pdat[ti] - tdat[ti]
    m = max(abs(err.min()), abs(err.max()), 1.0)
    im = ax[2].imshow(err, origin="lower", cmap="RdBu_r", vmin=-m, vmax=m,
                      extent=extent, aspect="equal")
    ax[2].set_title("Error (pred - true)"); ax[2].set_xlabel(hlabel); ax[2].set_ylabel("Z (m)")
    plt.colorbar(im, ax=ax[2], shrink=0.85, label="°C")
    fig.suptitle(f"t = {ti * DT_SLCF:.0f} s", fontsize=13)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def _uq_png(mean, std, true, slice_name, path):
    """4-panel UQ figure: truth / predictive mean / predictive std / standardized error."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pysmokview.viz.colormaps import get_colormap_config

    cm = get_colormap_config("TEMPERATURE")
    ti = N_TIMESTEPS // 2
    md, z, h, geom = _to_physical(mean, slice_name)
    sd, _, _, _ = _to_physical(std, slice_name)
    td, _, _, _ = _to_physical(true, slice_name)
    extent = [h[0], h[-1], z[0], z[-1]]
    hlabel = f"{geom['horiz_axis'].upper()} (m)"
    vmax = float(max(np.percentile(true, 99), np.percentile(mean, 99), T_AMBIENT + 1))
    zscore = np.abs(td[ti] - md[ti]) / np.maximum(sd[ti], 1e-6)

    fig, ax = plt.subplots(1, 4, figsize=(18, 6))
    panels = [(td[ti], "Ground truth (FDS)", cm["cmap"], T_AMBIENT, vmax, "°C"),
              (md[ti], "Predictive mean", cm["cmap"], T_AMBIENT, vmax, "°C"),
              (sd[ti], "Predictive std (uncertainty)", "viridis", 0, float(np.percentile(std, 99) + 1), "°C"),
              (zscore, "|error| / std", "magma", 0, 3, "σ")]
    for a, (fld, ttl, cmap, vmn, vmx, unit) in zip(ax, panels):
        im = a.imshow(fld, origin="lower", cmap=cmap, vmin=vmn, vmax=vmx,
                      extent=extent, aspect="equal")
        a.set_title(ttl); a.set_xlabel(hlabel); a.set_ylabel("Z (m)")
        plt.colorbar(im, ax=a, shrink=0.8, label=unit)
    fig.suptitle(f"{slice_name}  ({geom['plane']} @ {geom['normal']}={geom['pos']} m)   "
                 f"t = {ti * DT_SLCF:.0f} s", fontsize=13)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def _render_uq():
    uq_dir = os.path.join(OUTPUT_DIR, "uq")
    files = sorted(glob.glob(os.path.join(uq_dir, "*.npz")))
    if not files:
        print(f"No UQ files in {uq_dir}. Run evaluate_uq.py first."); return
    out = os.path.join(OUTPUT_DIR, "smokeview_uq"); os.makedirs(out, exist_ok=True)
    for fp in files:
        d = np.load(fp, allow_pickle=True)
        sname = str(d["slice"]); chid = str(d["chid"])
        print(f"  UQ overlay {chid} / {sname} ...")
        _uq_png(d["mean"], d["std"], d["true"], sname,
                os.path.join(out, f"{chid}__{sname}_uq.png"))
    print(f"\n  UQ overlays -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mp4", action="store_true", help="render MP4 instead of GIF (needs ffmpeg)")
    ap.add_argument("--uq", action="store_true", help="render generative-UQ overlays from outputs/uq/")
    args = ap.parse_args()

    if args.uq:
        _render_uq(); return

    fields_dir = os.path.join(OUTPUT_DIR, "fields")
    files = sorted(glob.glob(os.path.join(fields_dir, "*.npz")))
    if not files:
        print(f"No field files in {fields_dir}. Run evaluate.py first."); return
    if not HAVE_PYSMOKVIEW:
        print(f"ERROR: could not import pysmokview ({_IMPORT_ERR}).")
        print(f"       Ensure the sibling package exists at {PYSMOKVIEW_ROOT}.")
        return

    out = os.path.join(OUTPUT_DIR, "smokeview"); os.makedirs(out, exist_ok=True)
    fmt = "mp4" if args.mp4 else "gif"
    cfg = Config(); cfg.show_obst = False  # no geometry overlay for the surrogate fields

    animator = SliceAnimator(cfg)
    for fp in files:
        d = np.load(fp, allow_pickle=True)
        pred, true = d["pred"], d["true"]
        sname = str(d["slice"]); chid = str(d["chid"])
        stem = os.path.join(out, f"{chid}__{sname}")
        print(f"  Rendering {chid} / {sname} ...")
        # shared display range so pred and truth are directly comparable
        vmax = float(max(np.percentile(true, 99), np.percentile(pred, 99), T_AMBIENT + 1))
        animator.animate(_slice_data(true, sname), f"{stem}_true.{fmt}",
                         cmap="inferno", vmin=T_AMBIENT, vmax=vmax, format=fmt)
        animator.animate(_slice_data(pred, sname), f"{stem}_pred.{fmt}",
                         cmap="inferno", vmin=T_AMBIENT, vmax=vmax, format=fmt)
        _cmp_png(pred, true, sname, f"{stem}_cmp.png")

    print(f"\n  Smokeview output -> {out}")


if __name__ == "__main__":
    main()
