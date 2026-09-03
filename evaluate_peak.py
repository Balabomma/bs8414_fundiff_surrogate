"""Re-run the FunDiff evaluation and add the peak statistic it never wrote.

`evaluate.py` reports R2, RMSE, MAE, MBE, MAPE, the 95th percentile and SSIM,
but no peak error, so Chapter 5 could not make the safety statement Chapter 4
makes. This script adds it without touching `evaluate.py`: it sets
`config.MODEL_DIR`, imports `evaluate` as a module, and reuses that module's own
`decode_full`, `metrics` and `frame_ssim` unaltered, so every number other than
the new one is computed by the project's own code.

Nothing is overwritten. Results go to the path given by --out, leaving each run's
archived `metrics.json` untouched.

    python evaluate_peak.py --model-dir models_part1_fundiff_r1 --out peak_r1.json

Peak error is per case and per plane: the maximum of the predicted field minus
the maximum of the true field, averaged over held-out cases. Negative is
under-prediction of the peak, the unsafe direction. `peak_err_abs` is the mean
absolute peak discrepancy, which does not let over- and under-predictions cancel.

The archived R2 and RMSE are re-read and printed beside the recomputed ones as a
faithfulness check: if the loop here has drifted from `evaluate.py`, they differ.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True,
                    help="run directory name, e.g. models_part1_fundiff_r1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)

    # Point config at the requested run BEFORE evaluate binds MODEL_DIR.
    import config
    config.MODEL_DIR = os.path.join(config.PROJECT_DIR, a.model_dir)

    import evaluate as EV
    from config import (DEVICE, SLICE_IDS, N_LATENT_TOKENS, EMBED_DIM,
                        N_SAMPLING_STEPS, N_COND_SAMPLES, INFERENCE_SEED,
                        FIELD_HEIGHT, FIELD_WIDTH)
    from dataset_part1 import build_datasets
    from data_processing.dataset import full_grid_coords
    from model.fae import FunctionAutoencoder
    from model.dit import sample_latent
    from model_part1 import Part1Surrogate

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    model_dir = config.MODEL_DIR
    print("  run: %s" % model_dir)

    fae_ckpt = torch.load(os.path.join(model_dir, "fae_best.pt"),
                          map_location=device, weights_only=False)
    fae = FunctionAutoencoder(config).to(device)
    fae.load_state_dict(fae_ckpt["fae_state"])
    fae.eval()
    norm_stats = fae_ckpt["norm_stats"]

    dit = Part1Surrogate(config).to(device)
    dit.load_state_dict(torch.load(os.path.join(model_dir, "dit_best.pt"),
                                   map_location=device, weights_only=False)["dit_state"])
    dit.eval()

    _, _, test_ds, _ = build_datasets()
    grid = full_grid_coords(device=device)

    recon, gen = {}, {}
    for i in range(len(test_ds)):
        field, params, slice_id, ambient, chid = test_ds[i]
        sname = SLICE_IDS[int(slice_id)]
        stats = norm_stats.get(sname, norm_stats["global"])
        mean, std = stats["mean"], stats["std"]

        field = field.to(device)
        amb = ambient.to(device).unsqueeze(0)
        p = params.to(device).unsqueeze(0)
        s = slice_id.to(device).unsqueeze(0)
        true_phys = field.cpu().numpy() * std + mean

        z_recon = fae.encode(field.unsqueeze(0))
        pr = EV.decode_full(fae, z_recon, amb, device, grid).cpu().numpy()
        recon.setdefault(sname, []).append((pr * std + mean, true_phys))

        g = torch.Generator(device=device).manual_seed(INFERENCE_SEED + i)
        zs = [sample_latent(dit, p, s, N_LATENT_TOKENS, EMBED_DIM,
                            n_steps=N_SAMPLING_STEPS, generator=g)
              for _ in range(N_COND_SAMPLES)]
        z_mean = torch.stack(zs).mean(0)
        pg = EV.decode_full(fae, z_mean, amb, device, grid).cpu().numpy()
        gen.setdefault(sname, []).append((pg * std + mean, true_phys))

        if (i + 1) % 20 == 0:
            print("    %d/%d" % (i + 1, len(test_ds)))

    def peak_stats(pairs):
        """Per case: predicted peak minus true peak, over the whole field."""
        d = [float(pr.max() - tr.max()) for pr, tr in pairs]
        return {
            "peak_err": float(np.mean(d)),
            "peak_err_abs": float(np.mean(np.abs(d))),
            "peak_err_sd": float(np.std(d, ddof=1)) if len(d) > 1 else float("nan"),
            "n_cases": len(d),
            "n_under": int(sum(1 for x in d if x < 0)),
        }

    def summarize(store):
        allp, allt, per, alld = [], [], {}, []
        for sname in SLICE_IDS:
            if sname not in store:
                continue
            preds = np.stack([x for x, _ in store[sname]])
            trues = np.stack([y for _, y in store[sname]])
            m = EV.metrics(preds, trues)
            pn = (preds - norm_stats[sname]["mean"]) / norm_stats[sname]["std"]
            tn = (trues - norm_stats[sname]["mean"]) / norm_stats[sname]["std"]
            m["ssim"] = EV.frame_ssim(pn.reshape(-1, FIELD_HEIGHT, FIELD_WIDTH),
                                      tn.reshape(-1, FIELD_HEIGHT, FIELD_WIDTH))
            m.update(peak_stats(store[sname]))
            per[sname] = m
            alld.extend(store[sname])
            allp.append(preds.ravel())
            allt.append(trues.ravel())
        gm = EV.metrics(np.concatenate(allp), np.concatenate(allt))
        gm.update(peak_stats(alld))
        return {"global": gm, "per_slice": per}

    results = {"fae_recon": summarize(recon), "dit_gen": summarize(gen)}
    results["model_dir"] = a.model_dir

    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)

    # faithfulness check against the archived metrics.json
    arch = os.path.join(model_dir, "metrics.json")
    print("\n  %-16s %9s %9s %9s %9s" % ("plane", "R2 new", "R2 arch", "RMSE new", "RMSE arch"))
    if os.path.isfile(arch):
        A = json.load(open(arch))
        for sname in SLICE_IDS:
            n = results["dit_gen"]["per_slice"].get(sname)
            o = A.get("dit_gen", {}).get("per_slice", {}).get(sname)
            if n and o:
                print("  %-16s %9.4f %9.4f %9.2f %9.2f"
                      % (sname, n["r2"], o["r2"], n["rmse"], o["rmse"]))
    else:
        print("  (no archived metrics.json to compare against)")

    d = results["dit_gen"]["global"]
    print("\n  DiT generation, pooled: peak_err %+.2f degC  |peak_err| %.2f  "
          "under-predicting in %d of %d cases"
          % (d["peak_err"], d["peak_err_abs"], d["n_under"], d["n_cases"]))
    print("  wrote %s" % a.out)


if __name__ == "__main__":
    main()
