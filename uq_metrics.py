"""Shared uncertainty-quantification metrics for the BS8414 surrogates.

Method-agnostic: every UQ source (FunDiff generative samples, deep-ensemble
member spread, MC/concrete dropout passes) produces a predictive **mean** and
**std** per pixel; these functions score how trustworthy that std is.

Implements the calibration diagnostics used in Chavare, "Uncertainty
Quantification for AI-Driven Crash Simulation Surrogates" (arXiv:2607.18294):
central-interval coverage vs nominal, sharpness, a reliability curve + its
miscalibration area (ECE-style), Gaussian NLL, and the error<->uncertainty
correlation (does the model know where it is wrong?).

All inputs are numpy arrays of matching shape, in physical units (deg C).
"""
import numpy as np
from scipy.stats import norm


def coverage(truth, mean, std, z=1.0):
    """Fraction of points within +/- z*std of the mean (central Gaussian interval)."""
    std = np.maximum(std, 1e-8)
    return float(np.mean(np.abs(truth - mean) <= z * std))


def sharpness(std):
    """Mean predictive std (deg C) — lower is sharper (only meaningful if calibrated)."""
    return float(np.mean(std))


def gaussian_nll(truth, mean, std):
    """Mean negative log-likelihood under a Gaussian predictive (lower is better)."""
    std = np.maximum(std, 1e-8)
    return float(np.mean(0.5 * np.log(2 * np.pi * std ** 2) + (truth - mean) ** 2 / (2 * std ** 2)))


def error_uncertainty_corr(truth, mean, std):
    """Pearson corr between |error| and predictive std across points (>0 = useful)."""
    err = np.abs(truth - mean).ravel()
    s = std.ravel()
    if err.std() < 1e-8 or s.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(err, s)[0, 1])


def reliability_curve(truth, mean, std, nominals=None):
    """Empirical coverage of central intervals at each nominal probability.

    Returns (nominals, empirical). For a well-calibrated Gaussian predictive the
    empirical coverage tracks the nominal (the diagonal).
    """
    if nominals is None:
        nominals = np.linspace(0.05, 0.95, 19)
    std = np.maximum(std, 1e-8)
    a = np.abs(truth - mean) / std
    emp = np.array([np.mean(a <= norm.ppf(0.5 + p / 2)) for p in nominals])
    return np.asarray(nominals), emp


def calibration_area(truth, mean, std, nominals=None):
    """ECE-style miscalibration: mean |empirical - nominal| over the reliability curve."""
    nominals, emp = reliability_curve(truth, mean, std, nominals)
    return float(np.mean(np.abs(emp - nominals)))


def summarize(truth, mean, std):
    """Full UQ report dict for one set of predictions."""
    return {
        "rmse": float(np.sqrt(np.mean((mean - truth) ** 2))),
        "coverage_1sigma": coverage(truth, mean, std, 1.0),   # ideal 0.683
        "coverage_2sigma": coverage(truth, mean, std, 2.0),   # ideal 0.954
        "sharpness_degC": sharpness(std),
        "gaussian_nll": gaussian_nll(truth, mean, std),
        "err_unc_corr": error_uncertainty_corr(truth, mean, std),
        "miscalibration_area": calibration_area(truth, mean, std),
    }


IDEAL = {"coverage_1sigma": 0.6827, "coverage_2sigma": 0.9545}


def format_report(tag, s):
    lines = [
        f"  {tag}",
        f"    RMSE                : {s['rmse']:.2f} degC",
        f"    Coverage @1sigma    : {s['coverage_1sigma']:.3f}  (ideal 0.683)",
        f"    Coverage @2sigma    : {s['coverage_2sigma']:.3f}  (ideal 0.954)",
        f"    Sharpness (mean std): {s['sharpness_degC']:.2f} degC",
        f"    Gaussian NLL        : {s['gaussian_nll']:.3f}",
        f"    Error<->unc corr    : {s['err_unc_corr']:.3f}  (>0 = useful)",
        f"    Miscalibration area : {s['miscalibration_area']:.4f}  (0 = perfect)",
    ]
    return "\n".join(lines)
