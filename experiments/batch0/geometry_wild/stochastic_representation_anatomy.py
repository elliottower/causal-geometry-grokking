"""Stochastic Representation Anatomy — Open Question 3 from minimal representation proof.

For stochastic operations (power, shifted_mult) and a control always-class op
(multiplication), run 20 seeds each. For each seed:
  1. Train the grokking model
  2. Check if it grokked (test_loss < 0.1)
  3. If grokked: fit DAS at k=2, measure IIA, equivariance, rotation matrix
     structure, Fourier mode identification
  4. If not grokked: fit DAS at k=2 anyway, measure how badly it fails
  5. Compare representation structure between grokked vs non-grokked outcomes

Key analyses:
  - What fraction of seeds grok for each stochastic op?
  - When grokking occurs, is the representation still a minimal faithful irrep?
    (measure rotation_error, equivariance)
  - Do different seeds that grok pick the SAME Fourier mode k, or different ones?
  - What does the non-grokked representation look like?
    (IIA, equivariance, rotation_error for failed seeds)
  - Is the spectral gap (Davis-Kahan) predictive of which seeds grok?

Usage:
    # Full run (3 ops, 20 seeds each, GPU)
    python -u experiments/batch6_atlas/geometry_wild/stochastic_representation_anatomy.py \\
        --operations power,shifted_mult,multiplication \\
        --das-steps 400 --device cuda --output-dir /workspace/results

    # Fewer seeds for quick test
    python -u experiments/batch6_atlas/geometry_wild/stochastic_representation_anatomy.py \\
        --operations multiplication --seeds 0,1,2 \\
        --das-steps 5 --device cpu --no-wandb \\
        --output-dir experiments/results
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grokking_nonlinear_hunt import (
    build_data,
    cache_pairs,
    equivariance_test,
    grassmann_distance,
    train_das,
    train_grokking_model,
)

from factorization_circuits.pipeline.utils.factor_das_kernel import site_resid

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

# Default epoch counts per operation.
EPOCH_DEFAULTS = {
    "multiplication": 40000,
    "subtraction": 25000,
    "division": 40000,
    "bitwise_xor": 25000,
    "cubic_sum": 25000,
    "sum_of_squares": 25000,
    "max_ab": 25000,
    "composite_addition": 15000,
    "power": 80000,
    "shifted_mult": 40000,
    "cubing": 25000,
    "squaring": 25000,
    "abs_diff": 25000,
    "polynomial": 60000,
    "affine": 25000,
}

# Operations that use a different p value.
P_OVERRIDES = {
    "composite_addition": 91,
}

P_DEFAULTS = {
    "composite_addition": 91,
}


def utc_ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Circle fitting ──

def _circle_residual(params, points):
    """Residual: sum of squared (distance_to_center - r) for all points."""
    cx, cy, r = params
    dists = np.sqrt((points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2)
    return np.sum((dists - r) ** 2)


def fit_circle(points):
    """Fit a circle (cx, cy, r) to a set of 2D points using Nelder-Mead.

    Returns (cx, cy, r, residual, circularity_score).
    """
    cx0 = points[:, 0].mean()
    cy0 = points[:, 1].mean()
    r0 = np.sqrt(((points[:, 0] - cx0) ** 2 + (points[:, 1] - cy0) ** 2).mean())

    result = minimize(_circle_residual, x0=[cx0, cy0, r0], args=(points,),
                       method="Nelder-Mead",
                       options={"maxiter": 10000, "xatol": 1e-8, "fatol": 1e-8})
    cx, cy, r = result.x
    r = abs(r)

    dists = np.sqrt((points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2)
    mean_deviation = np.mean(np.abs(dists - r))
    circularity = 1.0 - mean_deviation / r if r > 1e-10 else 0.0

    return cx, cy, r, float(result.fun), circularity


# ── Fourier analysis ──

def dominant_fourier_mode(centroids_2d, p):
    """Find the dominant Fourier frequency in the centroid arrangement.

    Treats centroid (x, y) as a complex number z = x + iy, ordered by label
    value 0..p-1. Computes the DFT and returns the frequency with the largest
    magnitude.

    Returns (dominant_freq, spectrum_dict).
    """
    z = centroids_2d[:, 0] + 1j * centroids_2d[:, 1]
    z = z - z.mean()

    fft_vals = np.fft.fft(z)
    magnitudes = np.abs(fft_vals)

    max_freq = len(centroids_2d) // 2
    spectrum = {}
    for freq in range(1, max_freq + 1):
        spectrum[freq] = float(magnitudes[freq])

    if spectrum:
        dominant_freq = max(spectrum, key=spectrum.get)
    else:
        dominant_freq = 0

    return dominant_freq, spectrum


# ── Rotation matrix analysis ──

def rotation_matrix_analysis(centroids_2d, p):
    """Measure how close the label->centroid map is to a rotation by 2*pi*k/p.

    For a minimal faithful representation, centroid[label] should lie on a circle
    and the mapping label -> label+1 should correspond to a fixed 2D rotation.

    Returns (rotation_error, rotation_angle).
    - rotation_error: mean Frobenius deviation of consecutive-label transforms
      from the best-fit rotation matrix.
    - rotation_angle: angle of the best-fit rotation.
    """
    n = centroids_2d.shape[0]
    if n < 3:
        return float("nan"), float("nan")

    # Center the centroids
    center = centroids_2d.mean(axis=0)
    centered = centroids_2d - center

    # Compute the best-fit rotation from label i to label (i+1) % n
    # For each consecutive pair, compute the 2x2 matrix mapping c[i] -> c[i+1]
    # via Procrustes: R = V @ U^T from SVD of sum(c[i+1] @ c[i]^T)
    cross = np.zeros((2, 2))
    for i in range(n):
        j = (i + 1) % n
        cross += np.outer(centered[j], centered[i])

    U, _, Vt = np.linalg.svd(cross)
    # Ensure proper rotation (det = +1)
    det_sign = np.linalg.det(U @ Vt)
    if det_sign < 0:
        Vt[-1, :] *= -1
    R_best = U @ Vt

    # Rotation angle from the best-fit rotation matrix
    rotation_angle = float(np.arctan2(R_best[1, 0], R_best[0, 0]))

    # Measure deviation: for each consecutive pair, how far is c[i+1] from R @ c[i]?
    errors = []
    for i in range(n):
        j = (i + 1) % n
        predicted = R_best @ centered[i]
        error = np.linalg.norm(centered[j] - predicted)
        errors.append(error)

    # Normalize by mean radius
    mean_radius = np.mean(np.linalg.norm(centered, axis=1))
    rotation_error = float(np.mean(errors) / mean_radius) if mean_radius > 1e-10 else float("nan")

    return rotation_error, rotation_angle


# ── Spectral gap computation ──

def compute_spectral_gap(cached, k, device):
    """Compute spectral gap of the activation difference matrix at rank k.

    Returns spectral_gap (s_k - s_{k+1}).
    """
    diffs = []
    for _bt, base_act, src_act, _label in cached:
        diffs.append((base_act - src_act).detach().cpu())
    D = torch.stack(diffs, dim=0).float()

    sv = torch.linalg.svdvals(D).numpy()

    if k < len(sv):
        spectral_gap = float(sv[k - 1] - sv[k])
    else:
        spectral_gap = float(sv[-1])

    return spectral_gap


# ── Centroid computation ──

@torch.no_grad()
def compute_centroids(model, dataset, labels, Q, layer, device):
    """Forward pass all data, project H @ Q into 2D, compute per-label centroids.

    Returns centroids array of shape (n_labels, 2) ordered by label value.
    """
    hook_name = f"blocks.{layer}.hook_resid_post"
    batch_size = 256
    all_projected = []
    all_labels = []

    for start in range(0, len(dataset), batch_size):
        batch = dataset[start:start + batch_size]
        batch_labels = labels[start:start + batch_size]
        _, cache = model.run_with_cache(batch, names_filter=[hook_name])
        H = cache[hook_name][:, -1, :]
        Z = H @ Q
        all_projected.append(Z.cpu().numpy())
        all_labels.append(batch_labels.cpu().numpy())

    projected = np.concatenate(all_projected, axis=0)
    label_arr = np.concatenate(all_labels, axis=0)

    unique_labels = sorted(set(label_arr.tolist()))
    centroids = np.zeros((len(unique_labels), 2))
    for i, lbl in enumerate(unique_labels):
        mask = label_arr == lbl
        centroids[i] = projected[mask].mean(axis=0)

    return centroids, unique_labels


# ── Single seed pipeline ──

def run_one_seed(operation, p, seed, device, n_epochs, das_steps, output_path):
    """Train model at given seed, fit DAS at k=2, measure all metrics.

    Appends one JSONL row to output_path and returns the result dict.
    """
    logger.info("[%s] %s seed=%d (p=%d, %d epochs)", utc_ts(), operation, seed, p, n_epochs)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dataset, labels, train_idx, test_idx = build_data(operation, p, device, seed=seed)

    model, cfg, training_data = train_grokking_model(
        p, device, n_epochs=n_epochs,
        dataset=dataset, labels=labels,
        train_idx=train_idx, test_idx=test_idx,
        seed=seed,
    )

    train_losses = training_data["train_losses"]
    test_losses = training_data["test_losses"]
    final_train_loss = train_losses[-1] if train_losses else float("inf")
    final_test_loss = test_losses[-1] if test_losses else float("inf")
    grokked = final_test_loss < 0.1
    logger.info("  train_loss=%.4f  test_loss=%.4f  %s",
                final_train_loss, final_test_loss,
                "GROKKED" if grokked else "NOT GROKKED")

    model.eval()
    d = model.cfg.d_model
    layer = 0
    site = site_resid(layer, d)

    # Cache pairs for DAS
    cached = cache_pairs(model, dataset, labels, train_idx, layer, device)
    if len(cached) < 10:
        logger.warning("  Too few valid pairs (%d) -- skipping seed %d", len(cached), seed)
        row = {
            "operation": operation, "seed": int(seed), "grokked": grokked,
            "final_train_loss": float(final_train_loss),
            "final_test_loss": float(final_test_loss),
            "iia_k2": 0.0, "equivariance": 0.0,
            "rotation_error": float("nan"), "dominant_fourier_k": 0,
            "rotation_angle": float("nan"), "spectral_gap": float("nan"),
            "circle_radius": float("nan"), "r_ratio": float("nan"),
            "timestamp": utc_ts(),
        }
        with open(output_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        return row

    # Spectral gap (Davis-Kahan) -- computed before DAS fit
    spectral_gap = compute_spectral_gap(cached, k=2, device=device)
    logger.info("  Spectral gap (k=2): %.4f", spectral_gap)

    # Fit DAS at k=2
    Q, iia = train_das(model, site, cached, 2, das_steps, device)
    logger.info("  DAS k=2: IIA=%.3f", iia)

    # Equivariance
    equiv = equivariance_test(model, site, Q, dataset, labels, device, p, operation)
    logger.info("  Equivariance: %.3f (%d/%d)",
                equiv["equivariant_fraction"], equiv["n_equivariant"], equiv["n_tested"])

    # Centroids in DAS space
    centroids, unique_labels = compute_centroids(model, dataset, labels, Q, layer, device)

    # Circle fit
    cx, cy, r_fitted, circle_residual, circularity = fit_circle(centroids)
    r_weil = math.sqrt(p)
    r_ratio = r_fitted / r_weil if r_weil > 0 else float("nan")
    logger.info("  Circle: r=%.4f  r/sqrt(p)=%.4f", r_fitted, r_ratio)

    # Fourier mode
    n_labels = max(unique_labels) + 1
    label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
    ordered_centroids = np.zeros((n_labels, 2))
    for lbl in unique_labels:
        ordered_centroids[lbl] = centroids[label_to_idx[lbl]]
    dominant_freq, spectrum = dominant_fourier_mode(ordered_centroids, p)
    logger.info("  Dominant Fourier mode: %d", dominant_freq)

    # Rotation matrix analysis
    rotation_error, rotation_angle = rotation_matrix_analysis(ordered_centroids, p)
    logger.info("  Rotation error: %.4f  angle: %.4f rad", rotation_error, rotation_angle)

    row = {
        "operation": operation,
        "seed": int(seed),
        "grokked": grokked,
        "final_train_loss": float(final_train_loss),
        "final_test_loss": float(final_test_loss),
        "iia_k2": float(iia),
        "equivariance": float(equiv["equivariant_fraction"]),
        "rotation_error": float(rotation_error),
        "dominant_fourier_k": int(dominant_freq),
        "rotation_angle": float(rotation_angle),
        "spectral_gap": float(spectral_gap),
        "circle_radius": float(r_fitted),
        "r_ratio": float(r_ratio),
        "timestamp": utc_ts(),
    }

    with open(output_path, "a") as f:
        f.write(json.dumps(row) + "\n")
    logger.info("  Row appended to %s", output_path)

    return row


# ── Summary computation ──

def compute_summary(results_by_op):
    """Compute per-operation and cross-operation summary statistics.

    Returns a dict suitable for writing as summary JSON.
    """
    summary = {}
    for op, rows in results_by_op.items():
        grokked = [r for r in rows if r["grokked"]]
        ungrokked = [r for r in rows if not r["grokked"]]

        grok_fraction = len(grokked) / len(rows) if rows else 0.0

        def _mean(lst, key):
            vals = [r[key] for r in lst if not (isinstance(r[key], float) and math.isnan(r[key]))]
            return float(np.mean(vals)) if vals else float("nan")

        mean_iia_grokked = _mean(grokked, "iia_k2")
        mean_iia_ungrokked = _mean(ungrokked, "iia_k2")
        mean_equiv_grokked = _mean(grokked, "equivariance")
        mean_equiv_ungrokked = _mean(ungrokked, "equivariance")
        mean_rot_err_grokked = _mean(grokked, "rotation_error")
        mean_rot_err_ungrokked = _mean(ungrokked, "rotation_error")
        mean_spectral_gap_grokked = _mean(grokked, "spectral_gap")
        mean_spectral_gap_ungrokked = _mean(ungrokked, "spectral_gap")

        # Fourier mode consistency: fraction of grokked seeds using the most
        # common dominant_fourier_k
        if grokked:
            freq_counts = Counter(r["dominant_fourier_k"] for r in grokked)
            most_common_count = freq_counts.most_common(1)[0][1]
            fourier_mode_consistency = most_common_count / len(grokked)
            modal_fourier_k = freq_counts.most_common(1)[0][0]
        else:
            fourier_mode_consistency = float("nan")
            modal_fourier_k = None

        summary[op] = {
            "n_seeds": len(rows),
            "n_grokked": len(grokked),
            "n_ungrokked": len(ungrokked),
            "grok_fraction": grok_fraction,
            "mean_iia_grokked": mean_iia_grokked,
            "mean_iia_ungrokked": mean_iia_ungrokked,
            "mean_equivariance_grokked": mean_equiv_grokked,
            "mean_equivariance_ungrokked": mean_equiv_ungrokked,
            "mean_rotation_error_grokked": mean_rot_err_grokked,
            "mean_rotation_error_ungrokked": mean_rot_err_ungrokked,
            "mean_spectral_gap_grokked": mean_spectral_gap_grokked,
            "mean_spectral_gap_ungrokked": mean_spectral_gap_ungrokked,
            "fourier_mode_consistency": fourier_mode_consistency,
            "modal_fourier_k": modal_fourier_k,
        }

    return summary


# ── Main ──

def main():
    parser = argparse.ArgumentParser(
        description="Stochastic Representation Anatomy -- OQ3 from minimal representation proof")
    parser.add_argument("--operations", required=True,
                        help="Comma-separated list of operations")
    parser.add_argument("--seeds",
                        default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19",
                        help="Comma-separated seeds (default: 20 seeds 0..19)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--das-steps", type=int, default=400)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results"))
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    operations = [op.strip() for op in args.operations.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    device = args.device
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "stochastic_representation_anatomy.jsonl"
    summary_path = output_dir / "stochastic_representation_anatomy_summary.json"

    use_wandb = not args.no_wandb
    wandb_run = None
    if use_wandb:
        import wandb
        wandb_run = wandb.init(
            project="SAELensCircuitPort - Experimental",
            entity="factorized-circuits",
            name=f"stochastic-anatomy-{len(operations)}ops-{len(seeds)}seeds",
            job_type="geometry_wild",
            tags=["geometry_wild", "stochastic_anatomy", "open_question_3"],
            config={
                "operations": operations,
                "seeds": seeds,
                "das_steps": args.das_steps,
            },
        )

    logger.info("[%s] Stochastic Representation Anatomy", utc_ts())
    logger.info("  Operations: %s", operations)
    logger.info("  Seeds: %s (%d total)", seeds, len(seeds))
    logger.info("  DAS steps: %d, device: %s", args.das_steps, device)

    # ── Run all (operation, seed) pairs ──
    results_by_op = {op: [] for op in operations}
    total_pairs = len(operations) * len(seeds)
    pair_idx = 0

    for operation in operations:
        p = P_OVERRIDES.get(operation, 113)
        n_epochs = EPOCH_DEFAULTS.get(operation, 25000)

        logger.info("\n[%s] ===== %s (p=%d, %d epochs) =====", utc_ts(), operation, p, n_epochs)

        for seed in tqdm(seeds, desc=f"{operation}"):
            pair_idx += 1
            logger.info("\n[%s] --- %s seed=%d (%d/%d) ---",
                        utc_ts(), operation, seed, pair_idx, total_pairs)

            row = run_one_seed(operation, p, seed, device, n_epochs, args.das_steps,
                               output_path)
            results_by_op[operation].append(row)

            if wandb_run:
                import wandb
                prefix = f"{operation}/seed_{seed}"
                wandb.log({
                    f"{prefix}/iia_k2": row["iia_k2"],
                    f"{prefix}/equivariance": row["equivariance"],
                    f"{prefix}/grokked": int(row["grokked"]),
                    f"{prefix}/rotation_error": row["rotation_error"],
                    f"{prefix}/spectral_gap": row["spectral_gap"],
                    f"{prefix}/dominant_fourier_k": row["dominant_fourier_k"],
                })

    # ── Summary ──
    logger.info("\n[%s] === Computing summary ===", utc_ts())
    summary = compute_summary(results_by_op)

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("  Summary written to %s", summary_path)

    if wandb_run:
        import wandb
        for op, s in summary.items():
            wandb.log({
                f"summary/{op}/grok_fraction": s["grok_fraction"],
                f"summary/{op}/mean_iia_grokked": s["mean_iia_grokked"],
                f"summary/{op}/mean_iia_ungrokked": s["mean_iia_ungrokked"],
                f"summary/{op}/fourier_mode_consistency": s["fourier_mode_consistency"],
                f"summary/{op}/mean_spectral_gap_grokked": s["mean_spectral_gap_grokked"],
                f"summary/{op}/mean_spectral_gap_ungrokked": s["mean_spectral_gap_ungrokked"],
            })

        artifact = wandb.Artifact("stochastic-anatomy-results", type="geometry_wild_results")
        artifact.add_file(str(output_path))
        artifact.add_file(str(summary_path))
        wandb_run.log_artifact(artifact)
        wandb_run.finish()

    # ── Print summary table ──
    print(f"\n{'=' * 100}")
    print("Stochastic Representation Anatomy -- Summary")
    print(f"{'=' * 100}")

    for op in operations:
        s = summary[op]
        rows = results_by_op[op]
        grokked = [r for r in rows if r["grokked"]]
        ungrokked = [r for r in rows if not r["grokked"]]

        print(f"\n--- {op} ---")
        print(f"  Grok fraction: {s['grok_fraction']:.2f} ({s['n_grokked']}/{s['n_seeds']})")
        print(f"  Fourier mode consistency: {s['fourier_mode_consistency']:.2f}"
              f"  (modal k={s['modal_fourier_k']})")
        print()
        print(f"  {'Metric':<30s} {'Grokked':>10s} {'Ungrokked':>10s}")
        print(f"  {'-'*30} {'-'*10} {'-'*10}")
        print(f"  {'IIA (k=2)':<30s} {s['mean_iia_grokked']:10.3f} {s['mean_iia_ungrokked']:10.3f}")
        print(f"  {'Equivariance':<30s} {s['mean_equivariance_grokked']:10.3f} {s['mean_equivariance_ungrokked']:10.3f}")
        print(f"  {'Rotation error':<30s} {s['mean_rotation_error_grokked']:10.4f} {s['mean_rotation_error_ungrokked']:10.4f}")
        print(f"  {'Spectral gap':<30s} {s['mean_spectral_gap_grokked']:10.4f} {s['mean_spectral_gap_ungrokked']:10.4f}")

        if grokked:
            freq_counts = Counter(r["dominant_fourier_k"] for r in grokked)
            print(f"\n  Grokked Fourier mode distribution: {dict(freq_counts.most_common())}")

    # ── Per-seed detail table ──
    print(f"\n{'=' * 100}")
    print("Per-seed detail")
    print(f"{'=' * 100}")
    print(f"  {'Op':<16s} {'Seed':>5s} {'Grok':>5s} {'IIA':>6s} {'Equiv':>6s} "
          f"{'RotErr':>7s} {'FourK':>6s} {'SpGap':>7s} {'r/sqP':>7s} {'TestL':>7s}")
    print(f"  {'-'*16} {'-'*5} {'-'*5} {'-'*6} {'-'*6} "
          f"{'-'*7} {'-'*6} {'-'*7} {'-'*7} {'-'*7}")

    for op in operations:
        for r in results_by_op[op]:
            grok_str = "YES" if r["grokked"] else "no"
            re_str = f"{r['rotation_error']:7.4f}" if not math.isnan(r["rotation_error"]) else "    nan"
            sg_str = f"{r['spectral_gap']:7.4f}" if not math.isnan(r["spectral_gap"]) else "    nan"
            rr_str = f"{r['r_ratio']:7.4f}" if not math.isnan(r["r_ratio"]) else "    nan"
            print(f"  {r['operation']:<16s} {r['seed']:5d} {grok_str:>5s} "
                  f"{r['iia_k2']:6.3f} {r['equivariance']:6.3f} "
                  f"{re_str} {r['dominant_fourier_k']:6d} "
                  f"{sg_str} {rr_str} {r['final_test_loss']:7.4f}")

    # ── Spectral gap predictiveness ──
    print(f"\n{'=' * 100}")
    print("Spectral gap as grokking predictor")
    print(f"{'=' * 100}")

    for op in operations:
        s = summary[op]
        if s["n_grokked"] > 0 and s["n_ungrokked"] > 0:
            grokked_gaps = [r["spectral_gap"] for r in results_by_op[op]
                            if r["grokked"] and not math.isnan(r["spectral_gap"])]
            ungrokked_gaps = [r["spectral_gap"] for r in results_by_op[op]
                              if not r["grokked"] and not math.isnan(r["spectral_gap"])]
            if grokked_gaps and ungrokked_gaps:
                g_mean = np.mean(grokked_gaps)
                u_mean = np.mean(ungrokked_gaps)
                ratio = g_mean / u_mean if u_mean > 1e-10 else float("inf")
                print(f"  {op}: grokked_gap={g_mean:.4f}  ungrokked_gap={u_mean:.4f}  "
                      f"ratio={ratio:.2f}")
            else:
                print(f"  {op}: insufficient data for gap comparison")
        elif s["n_grokked"] == s["n_seeds"]:
            print(f"  {op}: all seeds grokked -- no contrast available")
        else:
            print(f"  {op}: no seeds grokked -- no contrast available")

    print(f"\n{'=' * 100}")
    logger.info("[%s] Done.", utc_ts())


if __name__ == "__main__":
    main()
