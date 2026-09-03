"""Prove every Part1 surrogate shares one data pipeline and one corpus.

A Part1 comparison is only attributable to the architecture if everything else is
provably identical. This checks both halves of that claim:

  1. **File parity** — the shared modules are byte-identical (SHA-256) across
     every project that uses them. Only `model_part1.py` may differ; that is the
     variable under test.

  2. **Data parity** — the arrays those modules produce hash the same, built
     inside each project's own venv. File parity alone would not catch a
     divergent corpus (a stale `PART1_SIMS_DIR`, cases added mid-run) or a
     different split assignment.

    python verify_parity_part1.py               # both checks
    python verify_parity_part1.py --data-only
    python verify_parity_part1.py --files-only  # fast, no subprocesses

Exit code is non-zero on any mismatch, so it can gate a run. Run it before every
Part1 comparison claim, and again after re-copying a shared file.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PROJECT_DIR)

# Thermocouple surrogates: 16 external TCs + the 5-channel HRR budget, driven by
# the shared regression trainer.
SENSOR_PROJECTS = ["bs8414_KAN_surrogate", "bs8414_MLP_surrogate",
                   "bs8414_surrogate_model", "bs8414_samba_mlp_surrogate",
                    "bs8414_KAN_v3_surrogate"]

# Slice-field surrogates: 5 planes x 128 x 64, driven by the shared slice trainer.
SLICE_PROJECTS = ["bs8414_slice_surrogate", "samba_MLP",
                  "bs8414_physicsnemo_surrogate",
                  "bs8414_physicsnemo_kan_surrogate"]

# Also on the Part1 slice corpus, but trains by rectified flow over FAE latents,
# so it shares the data layer and the conditioner and brings its own two-stage
# trainer (`dataset_part1.py` adapts the corpus to its dataset contract).
DIFFUSION_PROJECTS = ["bs8414_fundiff_surrogate",
                      "bs8414_fundiff_kan_surrogate"]

# KAN architecture variants: same corpus, same shared pipeline, the B-spline
# conditioning path is the one variable under test against their parents.
KAN_VARIANT_PROJECTS = ["bs8414_physicsnemo_kan_surrogate",
                        "bs8414_fundiff_kan_surrogate",
                        "bs8414_KAN_v3_surrogate"]

ALL_PROJECTS = SENSOR_PROJECTS + SLICE_PROJECTS + DIFFUSION_PROJECTS

# file -> the projects it must be identical across
SHARED = {
    "config_part1.py": ALL_PROJECTS,
    "data_loader_part1.py": ALL_PROJECTS,
    "physics_part1.py": ALL_PROJECTS,
    "explain_part1.py": ALL_PROJECTS,
    "causal_part1.py": ALL_PROJECTS,
    "train_part1.py": SENSOR_PROJECTS,
    "evaluate_part1.py": SENSOR_PROJECTS,
    "slice_loader_part1.py": SLICE_PROJECTS + DIFFUSION_PROJECTS,
    "part1_conditioning.py": SLICE_PROJECTS + DIFFUSION_PROJECTS,
    "slice_losses_part1.py": SLICE_PROJECTS,
    "train_slices_part1.py": SLICE_PROJECTS,
    "evaluate_slices_part1.py": SLICE_PROJECTS,
    # Vendored B-spline block, shared by the KAN architecture variants only.
    "kan_layers_part1.py": KAN_VARIANT_PROJECTS,
}

# Intentionally different in every project — this is the one variable.
VARIABLE_FILES = ["model_part1.py"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def check_files():
    print("  file parity (SHA-256)")
    ok = True
    for name, projects in SHARED.items():
        digests = {}
        for project in projects:
            path = os.path.join(ROOT, project, name)
            digests[project] = sha256_file(path) if os.path.isfile(path) else None

        present = {p: d for p, d in digests.items() if d}
        missing = [p for p, d in digests.items() if not d]
        unique = set(present.values())

        if missing:
            ok = False
            print(f"    {name:<26} MISSING in {', '.join(missing)}")
        elif len(unique) == 1:
            print(f"    {name:<26} OK   {next(iter(unique))[:16]}  "
                  f"({len(present)} projects)")
        else:
            ok = False
            print(f"    {name:<26} MISMATCH")
            for project, digest in present.items():
                print(f"      {digest[:16]}  {project}")
            print(f"      -> re-copy from bs8414_KAN_surrogate / "
                  f"bs8414_slice_surrogate; never hand-edit a copy")

    print("  variable file (expected to differ)")
    for name in VARIABLE_FILES:
        for project in ALL_PROJECTS:
            path = os.path.join(ROOT, project, name)
            digest = sha256_file(path)[:16] if os.path.isfile(path) else "ABSENT"
            print(f"    {digest}  {project}/{name}")
    return ok


SENSOR_FINGERPRINT = r"""
import hashlib, json, numpy as np
from data_loader_part1 import build_dataset, split_indices
params, tc, hrr, mask, meta, names = build_dataset(verbose=False)
def h(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float32).tobytes()).hexdigest()[:16]
idx = split_indices(meta)
print("FINGERPRINT " + json.dumps({
    "n_sims": int(len(params)),
    "params": h(params), "tc": h(tc), "hrr": h(hrr), "mask": h(mask),
    "chids": hashlib.sha256("|".join(m["chid"] for m in meta).encode()).hexdigest()[:16],
    "sensors": hashlib.sha256("|".join(names).encode()).hexdigest()[:16],
    "split": {k: len(v) for k, v in idx.items()},
}))
"""

SLICE_FINGERPRINT = r"""
import hashlib, json, numpy as np
from slice_loader_part1 import load_corpus
from data_loader_part1 import assign_split
params, fields, time_mask, plane_mask, meta = load_corpus(verbose=False)
def h(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]
counts = {"train": 0, "valid": 0, "test": 0}
for m in meta:
    counts[assign_split(m)] += 1
print("FINGERPRINT " + json.dumps({
    "n_sims": int(len(params)),
    "params": h(params), "fields": h(fields),
    "time_mask": h(time_mask), "plane_mask": h(plane_mask),
    "chids": hashlib.sha256("|".join(m["chid"] for m in meta).encode()).hexdigest()[:16],
    "n_planes": int(plane_mask.sum()),
    "split": counts,
}))
"""


def fingerprint(project, snippet):
    project_dir = os.path.join(ROOT, project)
    python = os.path.join(project_dir, "venv", "Scripts", "python.exe")
    if not os.path.isfile(python):
        return None, f"no venv at {python}"

    proc = subprocess.run([python, "-c", snippet], cwd=project_dir,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        tail = proc.stderr.strip().splitlines()
        return None, tail[-1] if tail else "failed"
    for line in proc.stdout.splitlines():
        if line.startswith("FINGERPRINT "):
            return json.loads(line[len("FINGERPRINT "):]), None
    return None, "no fingerprint emitted"


def check_group(label, projects, snippet, keys):
    print(f"\n  {label}")
    results = {}
    for project in projects:
        fp, err = fingerprint(project, snippet)
        if fp is None:
            print(f"    {project:<30} ERROR: {err}")
        else:
            summary = "  ".join(f"{k}={fp[k]}" for k in keys if k in fp)
            print(f"    {project:<30} n={fp['n_sims']}  {summary}")
        results[project] = fp

    good = {p: f for p, f in results.items() if f}
    if len(good) < len(projects):
        return False
    reference = good[projects[0]]
    ok = True
    for project, fp in good.items():
        if fp != reference:
            ok = False
            differing = [k for k in reference if fp.get(k) != reference[k]]
            print(f"    MISMATCH {project}: {', '.join(differing)}")
    if ok:
        print(f"    all {len(good)} agree — {reference['n_sims']} sims, "
              f"identical arrays and split {reference['split']}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-only", action="store_true")
    ap.add_argument("--files-only", action="store_true")
    args = ap.parse_args()

    print("=" * 78)
    print("  Part1 pipeline parity")
    print("=" * 78)

    ok = True
    if not args.data_only:
        ok &= check_files()
    if not args.files_only:
        ok &= check_group("thermocouple corpus (arrays built in each own venv)",
                          SENSOR_PROJECTS, SENSOR_FINGERPRINT,
                          ["params", "tc", "hrr"])
        ok &= check_group("slice corpus (arrays built in each own venv)",
                          SLICE_PROJECTS + DIFFUSION_PROJECTS, SLICE_FINGERPRINT,
                          ["params", "fields", "n_planes"])

    print("\n  " + ("PARITY HOLDS — Part1 results are comparable across projects"
                    if ok else
                    "PARITY BROKEN — do not compare results until this passes"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
