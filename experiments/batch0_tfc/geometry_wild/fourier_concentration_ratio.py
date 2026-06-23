"""Fourier Concentration Ratio (Theorem 14) — predicts grokking from algebra alone.

No training needed. For each operation f: Z/p x Z/p -> Z/p, compute the 2D DFT
of the operation table and measure how concentrated the spectral power is.

Claim: Operations that grok have R ≈ 1 (all power in one Fourier mode).
Operations that never grok have R ≈ 1/p (power spread uniformly).

This is a purely algebraic predictor — it requires NO trained model, NO GPU,
NO interventions. If it separates the three classes perfectly, it's the simplest
possible grokking predictor and a theorem about the operation, not the network.

For each operation:
  1. Build the p x p operation table: T[a,b] = f(a,b)
  2. Compute 2D DFT: F = fft2(T) / p^2
  3. Power spectrum: P[k,l] = |F[k,l]|^2
  4. Diagonal power (addition structure): D[k] = P[k,k]
  5. Anti-diagonal power (subtraction structure): A[k] = P[k,-k]
  6. Row/col marginal power
  7. Concentration ratio R = max_mode / total_power
  8. Also: effective rank of the power spectrum (how many modes needed for 90%)

Predicted relationship:
  - always-grok (multiplication, addition, division, ...): R close to 1
  - stochastic (composite_addition): R moderate
  - never-grok (power, floor_div): R close to 1/p

Also validates against the empirical grok rates from the 50-seed spectral gap
experiments (Experiment 9).

Usage:
    # Full run (CPU only — no training!)
    python -u experiments/batch6_atlas/geometry_wild/fourier_concentration_ratio.py \\
        --operations multiplication,subtraction,division,squaring,cubing,max_ab,abs_diff,sum_of_squares,power,shifted_mult,min_ab,floor_div,bitwise_xor,composite_addition \\
        --output-dir /workspace/results

    # Local test
    python -u experiments/batch6_atlas/geometry_wild/fourier_concentration_ratio.py \\
        --operations multiplication,subtraction --p 17
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

P_DEFAULTS = {"composite_addition": 91, "affine": 97}

EMPIRICAL_GROK_RATES = {
    "multiplication": 1.0, "subtraction": 1.0, "division": 1.0,
    "squaring": 1.0, "cubing": 1.0, "shifted_mult": 1.0,
    "max_ab": 1.0, "min_ab": 1.0, "bitwise_xor": 1.0,
    "gcd": 1.0, "abs_diff": 1.0, "sum_of_squares": 1.0,
    "composite_addition": 0.9,
    "power": 0.0, "floor_div": 0.0,
}

GROK_CLASS = {
    "multiplication": "always", "subtraction": "always", "division": "always",
    "squaring": "always", "cubing": "always", "shifted_mult": "always",
    "max_ab": "always", "min_ab": "always", "bitwise_xor": "always",
    "gcd": "always", "abs_diff": "always", "sum_of_squares": "always",
    "composite_addition": "stochastic",
    "power": "never", "floor_div": "never",
}


def utc_ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _op_fn(operation, a, b, p):
    """Compute operation(a, b) mod p."""
    if operation == "multiplication":
        return (a * b) % p
    elif operation == "subtraction":
        return (a - b) % p
    elif operation == "division":
        if b == 0:
            return 0
        return (a * pow(b, p - 2, p)) % p
    elif operation == "composite_addition":
        return (a + b) % p
    elif operation == "squaring":
        return (a * a) % p
    elif operation == "cubing":
        return (a * a * a) % p
    elif operation == "max_ab":
        return max(a, b) % p
    elif operation == "min_ab":
        return min(a, b) % p
    elif operation == "abs_diff":
        return abs(a - b) % p
    elif operation == "sum_of_squares":
        return (a * a + b * b) % p
    elif operation == "power":
        if a == 0:
            return 0
        return pow(a, b, p)
    elif operation == "shifted_mult":
        return ((a + 1) * (b + 1) - 1) % p
    elif operation == "floor_div":
        if b == 0:
            return 0
        return (a // b) % p
    elif operation == "bitwise_xor":
        return (a ^ b) % p
    elif operation == "gcd":
        from math import gcd
        return gcd(a, b) % p
    else:
        return None


def build_operation_table(operation, p):
    """Build the p x p operation table T[a,b] = op(a,b) mod p."""
    test = _op_fn(operation, 1, 1, p)
    if test is None:
        logger.warning("  Operation %s not recognized, skipping", operation)
        return None

    T = np.zeros((p, p), dtype=np.float64)
    for a in range(p):
        for b in range(p):
            T[a, b] = _op_fn(operation, a, b, p)
    return T


def fourier_analysis_2d(T, p):
    """Full 2D Fourier analysis of operation table.

    Returns dict with concentration metrics.
    """
    # 2D DFT
    F = np.fft.fft2(T) / p
    power = np.abs(F) ** 2

    total_power = np.sum(power)
    if total_power < 1e-12:
        return {"error": "zero_power"}

    # Max single mode
    max_power = np.max(power)
    max_idx = np.unravel_index(np.argmax(power), power.shape)

    # Concentration ratio: max mode / total
    R_max = float(max_power / total_power)

    # Diagonal concentration: modes where l = k (addition-like: f depends on a+b)
    diag_power = np.array([power[k, k] for k in range(p)])
    diag_total = np.sum(diag_power)
    R_diag = float(np.max(diag_power) / total_power) if total_power > 0 else 0.0
    diag_dominant_k = int(np.argmax(diag_power))

    # Theorem 14 formulation: concentration WITHIN diagonal modes only
    # R_thm14 = max(|F_diag[k]|²) / sum(|F_diag[k]|²)
    # This is the key predictor: R≈1 means one mode dominates (grokking)
    # R≈1/p means uniform spread (no grokking)
    R_thm14 = float(np.max(diag_power) / diag_total) if diag_total > 1e-12 else 0.0

    # Anti-diagonal: modes where l = -k mod p (subtraction-like: f depends on a-b)
    antidiag_power = np.array([power[k, (-k) % p] for k in range(p)])
    antidiag_total = np.sum(antidiag_power)
    R_antidiag = float(np.max(antidiag_power) / total_power)

    # Row-only: modes where l = 0 (f depends only on a)
    row_power = power[:, 0]
    R_row = float(np.max(row_power) / total_power)

    # Col-only: modes where k = 0 (f depends only on b)
    col_power = power[0, :]
    R_col = float(np.max(col_power) / total_power)

    # Effective rank: how many modes for 90% of power
    flat_power = power.flatten()
    sorted_power = np.sort(flat_power)[::-1]
    cumsum = np.cumsum(sorted_power)
    n_modes_90 = int(np.searchsorted(cumsum, 0.9 * total_power) + 1)
    n_modes_99 = int(np.searchsorted(cumsum, 0.99 * total_power) + 1)

    # Spectral entropy
    p_norm = flat_power / total_power
    p_norm = p_norm[p_norm > 1e-15]
    entropy = float(-np.sum(p_norm * np.log2(p_norm)))
    max_entropy = np.log2(p * p)

    # Best structure type
    R_best = max(R_diag, R_antidiag, R_row, R_col, R_max)
    if R_diag >= R_antidiag and R_diag >= R_row and R_diag >= R_col:
        structure = "additive"
    elif R_antidiag >= R_row and R_antidiag >= R_col:
        structure = "subtractive"
    elif R_row >= R_col:
        structure = "row_only"
    else:
        structure = "col_only"

    return {
        "R_thm14": R_thm14,
        "R_max": R_max,
        "R_diag": R_diag,
        "R_antidiag": R_antidiag,
        "R_row": R_row,
        "R_col": R_col,
        "R_best": float(R_best),
        "structure_type": structure,
        "max_mode": [int(x) for x in max_idx],
        "diag_dominant_k": int(diag_dominant_k),
        "n_modes_90pct": int(n_modes_90),
        "n_modes_99pct": int(n_modes_99),
        "spectral_entropy": entropy,
        "max_entropy": float(max_entropy),
        "normalized_entropy": float(entropy / max_entropy),
        "diag_fraction": float(diag_total / total_power),
        "antidiag_fraction": float(antidiag_total / total_power),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fourier Concentration Ratio — algebraic grokking predictor (no training)")
    parser.add_argument("--operations", required=True,
                        help="Comma-separated list of operations")
    parser.add_argument("--p", type=int, default=None)
    parser.add_argument("--das-steps", type=int, default=400,
                        help="Unused — kept for CLI compatibility with modal_atlas")
    parser.add_argument("--device", default="cpu",
                        help="Unused — this runs on CPU only")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results"))
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    operations = [op.strip() for op in args.operations.split(",")]
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "fourier_concentration_ratio.jsonl"

    logger.info("[%s] Fourier Concentration Ratio (Theorem 14)", utc_ts())
    logger.info("  Operations: %s", operations)
    logger.info("  Output: %s", output_path)

    results = []
    for op in tqdm(operations, desc="operations"):
        p = args.p or P_DEFAULTS.get(op, 113)
        logger.info("[%s] === %s (p=%d) ===", utc_ts(), op, p)

        T = build_operation_table(op, p)
        if T is None:
            continue

        analysis = fourier_analysis_2d(T, p)
        if "error" in analysis:
            logger.warning("  Skipping %s: %s", op, analysis["error"])
            continue

        grok_class = GROK_CLASS.get(op, "unknown")
        empirical_rate = EMPIRICAL_GROK_RATES.get(op, None)

        row = {
            "operation": op,
            "p": int(p),
            "grok_class": grok_class,
            "empirical_grok_rate": empirical_rate,
            **analysis,
            "timestamp": utc_ts(),
        }

        results.append(row)
        with open(output_path, "a") as f:
            f.write(json.dumps(row) + "\n")

        logger.info("  R_thm14=%.4f  R_max=%.4f  R_best=%.4f  structure=%s  "
                    "n_modes_90=%-3d  grok_class=%s",
                    analysis["R_thm14"], analysis["R_max"], analysis["R_best"],
                    analysis["structure_type"], analysis["n_modes_90pct"],
                    grok_class)

    # Summary and class separation analysis
    if results:
        logger.info("\n" + "=" * 80)
        logger.info("SUMMARY — Fourier Concentration Ratio vs Grokking Class")
        logger.info("=" * 80)

        by_class = {"always": [], "stochastic": [], "never": [], "unknown": []}
        for r in results:
            by_class[r["grok_class"]].append(r)

        for cls in ["always", "stochastic", "never"]:
            if by_class[cls]:
                R14_vals = [r["R_thm14"] for r in by_class[cls]]
                R_vals = [r["R_best"] for r in by_class[cls]]
                ent_vals = [r["normalized_entropy"] for r in by_class[cls]]
                n90_vals = [r["n_modes_90pct"] for r in by_class[cls]]
                logger.info("  %s (%d ops):", cls.upper(), len(by_class[cls]))
                logger.info("    R_thm14: %.4f ± %.4f  (range [%.4f, %.4f])",
                            np.mean(R14_vals), np.std(R14_vals),
                            np.min(R14_vals), np.max(R14_vals))
                logger.info("    R_best:  %.4f ± %.4f  (range [%.4f, %.4f])",
                            np.mean(R_vals), np.std(R_vals),
                            np.min(R_vals), np.max(R_vals))
                logger.info("    Entropy: %.4f ± %.4f", np.mean(ent_vals), np.std(ent_vals))
                logger.info("    N_modes_90%%: %.1f ± %.1f", np.mean(n90_vals), np.std(n90_vals))
                for r in by_class[cls]:
                    logger.info("      %-20s R14=%.4f  R=%.4f  struct=%-12s modes90=%d",
                                r["operation"], r["R_thm14"], r["R_best"],
                                r["structure_type"], r["n_modes_90pct"])

        # Separation test using R_thm14 (the theorem-specified metric)
        always_R14 = [r["R_thm14"] for r in by_class["always"]]
        never_R14 = [r["R_thm14"] for r in by_class["never"]]

        if always_R14 and never_R14:
            separation = np.min(always_R14) - np.max(never_R14)
            auc = float(sum(1 for a in always_R14 for n in never_R14 if a > n) /
                        (len(always_R14) * len(never_R14)))
            logger.info("\n  CLASS SEPARATION (R_thm14 — diagonal-mode concentration):")
            logger.info("    min(always R_thm14) = %.4f", np.min(always_R14))
            logger.info("    max(never R_thm14)  = %.4f", np.max(never_R14))
            logger.info("    Gap:                  %.4f", separation)
            logger.info("    AUC(always > never): %.3f", auc)
            logger.info("    THEOREM SUPPORTED: %s",
                        "YES" if separation > 0 else "NO (overlap)")

        # Correlation with empirical grok rate
        rates = [r["empirical_grok_rate"] for r in results if r["empirical_grok_rate"] is not None]
        R14_vals = [r["R_thm14"] for r in results if r["empirical_grok_rate"] is not None]
        if len(rates) > 3:
            corr = np.corrcoef(R14_vals, rates)[0, 1]
            logger.info("    Pearson(R_thm14, grok_rate): %.4f", corr)

        # Write summary
        summary = {
            "n_operations": len(results),
            "always_mean_R_thm14": float(np.mean(always_R14)) if always_R14 else None,
            "never_mean_R_thm14": float(np.mean(never_R14)) if never_R14 else None,
            "separation": float(separation) if always_R14 and never_R14 else None,
            "auc": float(auc) if always_R14 and never_R14 else None,
            "theorem_supported": bool(always_R14 and never_R14 and separation > 0),
            "correlation": float(corr) if len(rates) > 3 else None,
            "timestamp": utc_ts(),
        }
        summary_path = out_dir / "fourier_concentration_ratio_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("\n  Summary: %s", summary_path)


if __name__ == "__main__":
    main()
