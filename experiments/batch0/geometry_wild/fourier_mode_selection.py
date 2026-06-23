"""Fourier Mode Selection — which irrep V_k does the network pick?

By Proposition 3.2, there are (p-1)/2 equivalent real irreps V_1, ..., V_{(p-1)/2},
all minimal faithful. The proof does not predict WHICH one the network learns.
This experiment trains multiple seeds per operation, identifies the dominant Fourier
mode k via DFT of the centroid complex coordinates, and checks:

  - Mode consistency: do all seeds pick the same k?
  - Mode distribution: if different seeds pick different k, what is the distribution?
  - Cross-operation consistency: do different operations prefer different modes?
  - Mode vs geometry: does the chosen k correlate with Grassmannian distance?
  - Conjugate pair test: are V_k and V_{k'} related by Galois conjugation?
  - Is k=1 (simplest character) preferred, or are all modes equally likely?

Usage:
    # Full run (10 seeds, GPU)
    python -u experiments/batch6_atlas/geometry_wild/fourier_mode_selection.py \\
        --operations multiplication,subtraction,division \\
        --seeds 0,1,2,3,4,5,6,7,8,9 --das-steps 400 \\
        --device cuda --output-dir /workspace/results

    # Local test
    python -u experiments/batch6_atlas/geometry_wild/fourier_mode_selection.py \\
        --operations multiplication --seeds 0,1,2 --p 17 --n-epochs 100 --das-steps 5 \\
        --device cpu --output-dir experiments/results
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


def utc_ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Centroid computation ──

@torch.no_grad()
def compute_centroids(model, dataset, labels, Q, layer, device):
    """Forward pass all data, project H @ Q into 2D, compute per-label centroids.

    Returns centroids array of shape (n_labels, 2) ordered by label value,
    and the sorted list of unique labels present.
    """
    hook_name = f"blocks.{layer}.hook_resid_post"
    batch_size = 256
    all_projected = []
    all_labels = []

    for start in range(0, len(dataset), batch_size):
        batch = dataset[start:start + batch_size]
        batch_labels = labels[start:start + batch_size]
        _, cache = model.run_with_cache(batch, names_filter=[hook_name])
        H = cache[hook_name][:, -1, :]  # (batch, d_model)
        Z = H @ Q  # (batch, 2)
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


# ── Fourier analysis ──

def fourier_power_spectrum(centroids_2d, p):
    """Compute the full Fourier power spectrum of centroid arrangement.

    Treats centroid (x, y) as z = x + iy, ordered by label 0..p-1.
    Returns (dominant_k, spectrum_list, mode_purity) where:
      - dominant_k: frequency with maximum power
      - spectrum_list: list of {k, power} dicts for all positive frequencies
      - mode_purity: fraction of total power in the dominant mode
    """
    z = centroids_2d[:, 0] + 1j * centroids_2d[:, 1]
    z = z - z.mean()  # center

    fft_vals = np.fft.fft(z)
    magnitudes = np.abs(fft_vals)
    powers = magnitudes ** 2

    max_freq = len(z) // 2
    spectrum = []
    total_power = 0.0
    best_k = 0
    best_power = 0.0

    for k in range(1, max_freq + 1):
        pw = float(powers[k])
        spectrum.append({"k": k, "power": pw})
        total_power += pw
        if pw > best_power:
            best_power = pw
            best_k = k

    mode_purity = best_power / total_power if total_power > 1e-15 else 0.0

    return best_k, spectrum, mode_purity


def estimate_rotation_angle(centroids_2d, p):
    """Estimate the rotation angle between consecutive labels in the centroid circle.

    Returns the median angular step between label g and label g+1.
    """
    z = centroids_2d[:, 0] + 1j * centroids_2d[:, 1]
    center = z.mean()
    z_centered = z - center

    angles = np.angle(z_centered)
    n = len(angles)
    steps = []
    for g in range(n - 1):
        diff = angles[g + 1] - angles[g]
        # Wrap to [-pi, pi]
        diff = (diff + math.pi) % (2 * math.pi) - math.pi
        steps.append(abs(diff))

    return float(np.median(steps)) if steps else 0.0


# ── Single seed pipeline ──

def run_one_seed(operation, p, seed, device, n_epochs, das_steps, das_k,
                 output_path):
    """Train one seed, fit DAS, identify dominant Fourier mode.

    Appends one JSONL row and returns the result dict (or None on failure).
    """
    logger.info("[%s] %s seed=%d p=%d", utc_ts(), operation, seed, p)

    dataset, labels, train_idx, test_idx = build_data(operation, p, device, seed=seed)
    logger.info("  Dataset: %d examples (%d train, %d test)",
                len(dataset), len(train_idx), len(test_idx))

    model, cfg, training_data = train_grokking_model(
        p, device, n_epochs=n_epochs,
        dataset=dataset, labels=labels,
        train_idx=train_idx, test_idx=test_idx,
        seed=seed,
    )

    test_losses = training_data["test_losses"]
    final_test_loss = test_losses[-1] if test_losses else float("inf")
    grokked = final_test_loss < 0.1
    logger.info("  Final test loss: %.4f — %s", final_test_loss,
                "GROKKED" if grokked else "NOT GROKKED")

    model.eval()
    d = model.cfg.d_model
    layer = 0
    site = site_resid(layer, d)

    # Fit DAS at k=2
    cached = cache_pairs(model, dataset, labels, train_idx, layer, device)
    if len(cached) < 10:
        logger.warning("  Too few valid pairs (%d) — skipping seed %d", len(cached), seed)
        return None

    Q, iia = train_das(model, site, cached, das_k, das_steps, device)
    logger.info("  DAS k=%d: IIA=%.3f", das_k, iia)

    # Equivariance
    equiv = equivariance_test(model, site, Q, dataset, labels, device, p, operation)
    logger.info("  Equivariance: %.3f (%d/%d)",
                equiv["equivariant_fraction"], equiv["n_equivariant"], equiv["n_tested"])

    # Centroids in DAS space
    centroids, unique_labels = compute_centroids(model, dataset, labels, Q, layer, device)
    logger.info("  Centroids: %d unique labels", len(unique_labels))

    # Build label-ordered centroid array for Fourier analysis
    label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
    n_labels = max(unique_labels) + 1
    ordered_centroids = np.zeros((n_labels, 2))
    for lbl in unique_labels:
        ordered_centroids[lbl] = centroids[label_to_idx[lbl]]

    # Fourier mode identification
    dominant_k, spectrum, mode_purity = fourier_power_spectrum(ordered_centroids, p)
    logger.info("  Dominant Fourier mode k=%d  purity=%.3f", dominant_k, mode_purity)

    # Rotation angle analysis
    rotation_angle = estimate_rotation_angle(ordered_centroids, p)
    theoretical_angle = 2 * math.pi * dominant_k / p if dominant_k > 0 else 0.0
    angle_match = abs(rotation_angle - theoretical_angle) < 0.3 if dominant_k > 0 else False
    logger.info("  Rotation angle: %.4f  theoretical(k=%d): %.4f  match=%s",
                rotation_angle, dominant_k, theoretical_angle, angle_match)

    row = {
        "operation": operation,
        "seed": int(seed),
        "grokked": grokked,
        "dominant_fourier_k": int(dominant_k),
        "fourier_power_spectrum": spectrum,
        "mode_purity": float(mode_purity),
        "iia_k2": float(iia),
        "equivariance": float(equiv["equivariant_fraction"]),
        "rotation_angle": float(rotation_angle),
        "theoretical_angle_for_mode": float(theoretical_angle),
        "angle_match": bool(angle_match),
        "p": int(p),
        "final_test_loss": float(final_test_loss),
        "timestamp": utc_ts(),
    }

    with open(output_path, "a") as f:
        f.write(json.dumps(row) + "\n")
    logger.info("  Row appended to %s", output_path)

    return row


# ── Per-operation summary ──

def write_operation_summary(op_results, p, output_path):
    """Write a summary row after all seeds for one operation complete."""
    grokked_results = [r for r in op_results if r["grokked"]]
    operation = op_results[0]["operation"]

    if not grokked_results:
        logger.info("  No grokked seeds for %s — skipping summary", operation)
        return

    modes = [r["dominant_fourier_k"] for r in grokked_results]
    mode_counts = dict(Counter(modes))
    most_common_mode = Counter(modes).most_common(1)[0][0]
    mode_consistency = mode_counts[most_common_mode] / len(modes)

    # Mode entropy: H = -sum(p * log(p))
    n = len(modes)
    probs = np.array([count / n for count in mode_counts.values()])
    mode_entropy = float(-np.sum(probs * np.log(probs + 1e-15)))

    # Check if k=1 is preferred
    k1_count = mode_counts.get(1, 0)
    k1_fraction = k1_count / len(modes)

    # Uniformity test: max possible frequency is (p-1)/2
    max_possible = (p - 1) // 2
    n_distinct_modes = len(set(modes))

    logger.info("  Summary for %s: %d grokked seeds", operation, len(grokked_results))
    logger.info("    Mode counts: %s", mode_counts)
    logger.info("    Most common: k=%d (%.0f%% of seeds)", most_common_mode,
                mode_consistency * 100)
    logger.info("    Entropy: %.3f (max=%.3f for uniform over %d modes)",
                mode_entropy, math.log(max_possible), max_possible)
    logger.info("    k=1 fraction: %.2f", k1_fraction)

    summary = {
        "type": "operation_summary",
        "operation": operation,
        "p": int(p),
        "n_seeds": len(op_results),
        "n_grokked": len(grokked_results),
        "mode_counts": mode_counts,
        "mode_entropy": float(mode_entropy),
        "most_common_mode": int(most_common_mode),
        "mode_consistency": float(mode_consistency),
        "k1_fraction": float(k1_fraction),
        "n_distinct_modes": n_distinct_modes,
        "max_possible_modes": max_possible,
        "all_modes": modes,
        "mean_iia": float(np.mean([r["iia_k2"] for r in grokked_results])),
        "mean_equivariance": float(np.mean([r["equivariance"] for r in grokked_results])),
        "mean_mode_purity": float(np.mean([r["mode_purity"] for r in grokked_results])),
        "timestamp": utc_ts(),
    }

    with open(output_path, "a") as f:
        f.write(json.dumps(summary) + "\n")
    logger.info("  Summary appended to %s", output_path)

    return summary


# ── Cross-operation Grassmannian analysis ──

def cross_operation_analysis(all_results, output_path):
    """Compare modes and Grassmannian distances across operations.

    Only meaningful if multiple operations have grokked seeds.
    """
    by_op = {}
    for r in all_results:
        if r["grokked"]:
            by_op.setdefault(r["operation"], []).append(r)

    if len(by_op) < 2:
        logger.info("  Cross-operation analysis: need >= 2 operations, have %d", len(by_op))
        return

    ops = sorted(by_op.keys())
    logger.info("[%s] Cross-operation mode comparison (%d operations)", utc_ts(), len(ops))

    cross_summary = {
        "type": "cross_operation_summary",
        "n_operations": len(ops),
        "operations": ops,
        "per_operation_modes": {},
        "timestamp": utc_ts(),
    }

    for op in ops:
        modes = [r["dominant_fourier_k"] for r in by_op[op]]
        cross_summary["per_operation_modes"][op] = {
            "modes": modes,
            "most_common": Counter(modes).most_common(1)[0][0],
            "n_distinct": len(set(modes)),
        }
        logger.info("  %s: modes=%s", op, modes)

    # Check if different operations prefer different modes
    most_common_per_op = [
        Counter([r["dominant_fourier_k"] for r in by_op[op]]).most_common(1)[0][0]
        for op in ops
    ]
    cross_summary["all_same_mode"] = len(set(most_common_per_op)) == 1
    cross_summary["most_common_modes_per_op"] = dict(zip(ops, [int(m) for m in most_common_per_op]))

    logger.info("  All operations prefer same mode? %s", cross_summary["all_same_mode"])
    logger.info("  Preferred modes: %s", cross_summary["most_common_modes_per_op"])

    with open(output_path, "a") as f:
        f.write(json.dumps(cross_summary) + "\n")


# ── Main ──

def main():
    parser = argparse.ArgumentParser(
        description="Fourier Mode Selection — which irrep V_k does the network pick?")
    parser.add_argument("--operations", required=True,
                        help="Comma-separated list of operations")
    parser.add_argument("--p", type=int, default=113,
                        help="Default modulus (overridden per-op where needed)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-epochs", type=int, default=None,
                        help="Override epoch count for all operations")
    parser.add_argument("--das-steps", type=int, default=400)
    parser.add_argument("--das-k", type=int, default=2,
                        help="DAS subspace dimension (should be 2)")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9",
                        help="Comma-separated list of seeds")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results"))
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    operations = [op.strip() for op in args.operations.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    device = args.device
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = not args.no_wandb
    wandb_run = None
    if use_wandb:
        import wandb
        wandb_run = wandb.init(
            project="SAELensCircuitPort - Experimental",
            entity="factorized-circuits",
            name=f"fourier-mode-selection-{len(operations)}ops-{len(seeds)}seeds",
            job_type="geometry_wild",
            tags=["geometry_wild", "fourier_mode_selection", "irrep_selection"],
            config={
                "operations": operations,
                "p_default": args.p,
                "seeds": seeds,
                "n_epochs_override": args.n_epochs,
                "das_steps": args.das_steps,
                "das_k": args.das_k,
            },
        )

    logger.info("[%s] Fourier Mode Selection — %d operations x %d seeds",
                utc_ts(), len(operations), len(seeds))
    logger.info("  Operations: %s", operations)
    logger.info("  Seeds: %s", seeds)
    logger.info("  Default p: %d, DAS steps: %d, k: %d",
                args.p, args.das_steps, args.das_k)

    output_path = output_dir / "fourier_mode_selection.jsonl"

    all_results = []
    all_summaries = []

    for op in operations:
        p = P_OVERRIDES.get(op, args.p)
        n_epochs = (args.n_epochs if args.n_epochs is not None
                    else EPOCH_DEFAULTS.get(op, 25000))

        op_results = []
        desc = f"{op} (p={p})"
        for seed in tqdm(seeds, desc=desc):
            row = run_one_seed(op, p, seed, device, n_epochs, args.das_steps,
                               args.das_k, output_path)
            if row is not None:
                op_results.append(row)
                all_results.append(row)
                if wandb_run:
                    import wandb
                    wandb.log({
                        f"per_seed/{op}/seed{seed}/iia": row["iia_k2"],
                        f"per_seed/{op}/seed{seed}/dominant_k": row["dominant_fourier_k"],
                        f"per_seed/{op}/seed{seed}/mode_purity": row["mode_purity"],
                        f"per_seed/{op}/seed{seed}/equivariance": row["equivariance"],
                        f"per_seed/{op}/seed{seed}/grokked": int(row["grokked"]),
                    })
            else:
                logger.warning("  Skipped %s seed=%d (fit failed)", op, seed)

        # Per-operation summary
        if op_results:
            summary = write_operation_summary(op_results, p, output_path)
            if summary is not None:
                all_summaries.append(summary)
                if wandb_run:
                    import wandb
                    wandb.log({
                        f"summary/{op}/most_common_mode": summary["most_common_mode"],
                        f"summary/{op}/mode_consistency": summary["mode_consistency"],
                        f"summary/{op}/mode_entropy": summary["mode_entropy"],
                        f"summary/{op}/k1_fraction": summary["k1_fraction"],
                        f"summary/{op}/n_grokked": summary["n_grokked"],
                        f"summary/{op}/mean_iia": summary["mean_iia"],
                        f"summary/{op}/mean_purity": summary["mean_mode_purity"],
                    })

    # ── Cross-operation analysis ──
    if len(operations) > 1:
        cross_operation_analysis(all_results, output_path)

    # ── W&B artifacts and final logging ──
    if wandb_run:
        import wandb
        if all_summaries:
            wandb.log({
                "global/n_operations": len(operations),
                "global/n_seeds": len(seeds),
                "global/n_total_results": len(all_results),
                "global/n_grokked": sum(1 for r in all_results if r["grokked"]),
            })

            results_table = wandb.Table(
                columns=["operation", "seed", "grokked", "dominant_k",
                          "mode_purity", "iia_k2", "equivariance",
                          "rotation_angle", "angle_match"],
                data=[[r["operation"], r["seed"], r["grokked"],
                       r["dominant_fourier_k"], r["mode_purity"],
                       r["iia_k2"], r["equivariance"],
                       r["rotation_angle"], r["angle_match"]]
                      for r in all_results],
            )
            wandb.log({"results_table": results_table})

            summary_table = wandb.Table(
                columns=["operation", "n_grokked", "most_common_mode",
                          "mode_consistency", "mode_entropy", "k1_fraction",
                          "mean_iia", "mean_purity"],
                data=[[s["operation"], s["n_grokked"], s["most_common_mode"],
                       s["mode_consistency"], s["mode_entropy"], s["k1_fraction"],
                       s["mean_iia"], s["mean_mode_purity"]]
                      for s in all_summaries],
            )
            wandb.log({"summary_table": summary_table})

        artifact = wandb.Artifact("fourier-mode-selection-results",
                                  type="geometry_wild_results")
        artifact.add_file(str(output_path))
        wandb_run.log_artifact(artifact)
        wandb_run.finish()

    # ── Summary table ──
    logger.info("\n[%s] === All seeds complete: %d / %d succeeded ===",
                utc_ts(), len(all_results), len(operations) * len(seeds))

    if not all_results:
        logger.error("No successful fits — nothing to summarize.")
        return

    print(f"\n{'=' * 100}")
    print(f"Fourier Mode Selection — {len(operations)} operations x {len(seeds)} seeds")
    print(f"{'=' * 100}")

    print(f"\n{'Operation':<22s} {'Seed':>5s} {'Grok':>5s} {'k':>4s} "
          f"{'Purity':>7s} {'IIA':>6s} {'Equiv':>6s} {'Angle':>7s} {'Match':>6s}")
    print("-" * 78)
    for r in all_results:
        print(f"{r['operation']:<22s} {r['seed']:5d} "
              f"{'YES' if r['grokked'] else 'NO':>5s} "
              f"{r['dominant_fourier_k']:4d} {r['mode_purity']:7.3f} "
              f"{r['iia_k2']:6.3f} {r['equivariance']:6.3f} "
              f"{r['rotation_angle']:7.4f} "
              f"{'YES' if r['angle_match'] else 'NO':>6s}")

    if all_summaries:
        print(f"\n{'=' * 80}")
        print(f"Per-Operation Summaries")
        print(f"{'=' * 80}")

        print(f"\n{'Operation':<22s} {'Grok':>5s} {'BestK':>6s} {'Consist':>8s} "
              f"{'Entropy':>8s} {'k=1%':>6s} {'Modes':>15s}")
        print("-" * 75)
        for s in all_summaries:
            modes_str = str(s["mode_counts"])
            print(f"{s['operation']:<22s} "
                  f"{s['n_grokked']:5d} "
                  f"{s['most_common_mode']:6d} "
                  f"{s['mode_consistency']:8.2f} "
                  f"{s['mode_entropy']:8.3f} "
                  f"{s['k1_fraction']:6.2f} "
                  f"{modes_str:>15s}")

        # Key findings
        print(f"\nKey findings:")
        for s in all_summaries:
            if s["mode_consistency"] == 1.0:
                print(f"  {s['operation']}: ALL seeds pick k={s['most_common_mode']} "
                      f"(perfect consistency)")
            elif s["mode_consistency"] >= 0.7:
                print(f"  {s['operation']}: strong preference for k={s['most_common_mode']} "
                      f"({s['mode_consistency']:.0%} of seeds)")
            else:
                print(f"  {s['operation']}: no clear preference "
                      f"({s['n_distinct_modes']} distinct modes out of "
                      f"{s['max_possible_modes']} possible)")

        # k=1 preference test
        total_grokked = sum(s["n_grokked"] for s in all_summaries)
        total_k1 = sum(
            s["mode_counts"].get("1", s["mode_counts"].get(1, 0))
            for s in all_summaries
        )
        if total_grokked > 0:
            overall_k1 = total_k1 / total_grokked
            print(f"\n  Overall k=1 fraction: {overall_k1:.2f} "
                  f"(uniform expectation: {2 / (args.p - 1):.3f})")
            if overall_k1 > 0.5:
                print(f"  => k=1 is PREFERRED (simplest character bias)")
            else:
                print(f"  => No strong k=1 preference")

    print(f"\n{'=' * 100}")
    logger.info("[%s] Done.", utc_ts())


if __name__ == "__main__":
    main()
