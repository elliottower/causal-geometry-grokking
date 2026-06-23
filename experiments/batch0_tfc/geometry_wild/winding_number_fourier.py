"""Winding Number Fourier (E8) — Theorem 2.3: winding number of DAS centroids == dominant Fourier mode.

For each grokked operation, computes:
  1. DAS k=2 projections of activations
  2. Per-class centroids in 2D DAS space: mu_c for c = 0,...,p-1
  3. Winding number of the centroid sequence around its mean
  4. Dominant Fourier mode index from embedding weights W_E
  5. Checks whether winding_number == dominant_fourier_mode

Prediction: >99% match across all grokked group-action operations and seeds.

Operations tested (always-class ops that reliably grok):
  multiplication, subtraction, division, composite_addition,
  cubic_sum, shifted_mult

Usage:
    # Full run (GPU, 10 seeds per op)
    python -u experiments/batch6_atlas/geometry_wild/winding_number_fourier.py \\
        --operations multiplication,subtraction,division,composite_addition,cubic_sum,shifted_mult \\
        --n-seeds 10 --device cuda --output-dir /workspace/results

    # Single operation test
    python -u experiments/batch6_atlas/geometry_wild/winding_number_fourier.py \\
        --operations subtraction --n-seeds 3 --device cuda --output-dir /workspace/results

    # Local test
    python -u experiments/batch6_atlas/geometry_wild/winding_number_fourier.py \\
        --operations subtraction,multiplication --n-seeds 2 --p 17 \\
        --n-epochs 100 --das-steps 5 --device cpu --output-dir experiments/results --no-wandb
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
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
    train_das,
    train_grokking_model,
)

from factorization_circuits.pipeline.utils.factor_das_kernel import site_resid

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

EPOCH_DEFAULTS = {
    "multiplication": 40000,
    "subtraction": 25000,
    "division": 40000,
    "cubic_sum": 25000,
    "shifted_mult": 25000,
    "composite_addition": 15000,
}

P_DEFAULTS = {
    "composite_addition": 91,
}

P_OVERRIDES = {
    "composite_addition": 91,
}

ALWAYS_CLASS_OPS = [
    "multiplication", "subtraction", "division",
    "composite_addition", "cubic_sum", "shifted_mult",
]


def utc_ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# -- Centroid computation --

@torch.no_grad()
def compute_centroids_2d(model, dataset, labels, Q, layer, device):
    """Project all activations into 2D DAS space, compute per-label centroids.

    Returns (centroids, unique_labels) where centroids is (n_labels, 2) numpy array
    ordered by label value.
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


# -- Winding number computation --

def compute_winding_number(centroids_2d):
    """Compute the winding number of the centroid sequence around its mean.

    For centroids mu_0, mu_1, ..., mu_{p-1} ordered by label value:
      1. Compute mean: mu_bar = mean(mu_c)
      2. Center: delta_c = mu_c - mu_bar
      3. Compute angle for each centroid: theta_c = atan2(delta_c[1], delta_c[0])
      4. Sum angular differences between consecutive centroids (wrapping to [-pi, pi])
      5. winding_number = round(total_angle / (2*pi))

    Returns (winding_number, total_angle).
    """
    center = centroids_2d.mean(axis=0)
    centered = centroids_2d - center
    n = len(centroids_2d)

    angles = np.arctan2(centered[:, 1], centered[:, 0])

    total_angle = 0.0
    for i in range(n):
        j = (i + 1) % n
        diff = angles[j] - angles[i]
        # Wrap to [-pi, pi]
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        total_angle += diff

    winding_number = round(total_angle / (2 * math.pi))
    return winding_number, total_angle


# -- Dominant Fourier mode from embedding weights --

def compute_dominant_fourier_mode(model, p):
    """Extract dominant Fourier mode index from embedding weights W_E.

    W_E has shape (vocab_size, d_model).  We use the first p rows (token indices
    0..p-1 corresponding to the p residue classes).

    For each column j of W_E:
      F[k, j] = sum_a W_E[a, j] * exp(-2*pi*i * k * a / p)

    Amplitude spectrum: A[k] = sum_j |F[k, j]|^2
    dominant_mode = argmax A[k] for k in 1..(p-1)//2

    Returns (dominant_mode, spectrum_dict).
    """
    W_E = model.embed.W_E.detach().cpu().numpy()  # (vocab_size, d_model)
    W_E_mod = W_E[:p, :]  # (p, d_model)

    # DFT along the token dimension
    # F[k, j] = sum_a W_E[a, j] * exp(-2*pi*i * k * a / p)
    a_vals = np.arange(p)  # (p,)
    max_freq = (p - 1) // 2

    spectrum = {}
    for k in range(1, max_freq + 1):
        # Complex exponential: exp(-2*pi*i * k * a / p) for each a
        phases = np.exp(-2j * math.pi * k * a_vals / p)  # (p,)
        # DFT coefficients: (d_model,) = phases^T @ W_E_mod
        F_k = phases @ W_E_mod  # (d_model,) complex
        amplitude = float(np.sum(np.abs(F_k) ** 2))
        spectrum[k] = amplitude

    dominant_mode = max(spectrum, key=spectrum.get) if spectrum else 0
    return dominant_mode, spectrum


# -- Single (operation, seed) pipeline --

def run_one_trial(operation, p, device, n_epochs, das_steps, seed, output_path):
    """Train model, fit DAS k=2, compute winding number and Fourier mode, compare.

    Appends one JSONL row and returns result dict.
    """
    logger.info("[%s] === %s (p=%d, seed=%d) ===", utc_ts(), operation, p, seed)

    dataset, labels, train_idx, test_idx = build_data(operation, p, device, seed=seed)
    logger.info("  Dataset: %d examples (%d train, %d test)",
                len(dataset), len(train_idx), len(test_idx))

    # Train grokking model
    logger.info("[%s]  Training grokking model (%d epochs)...", utc_ts(), n_epochs)
    model, cfg, training_data = train_grokking_model(
        p, device, n_epochs=n_epochs,
        dataset=dataset, labels=labels,
        train_idx=train_idx, test_idx=test_idx,
        seed=seed,
    )

    test_losses = training_data["test_losses"]
    final_test_loss = test_losses[-1] if test_losses else float("inf")
    grokked = final_test_loss < 0.1
    logger.info("  Final test loss: %.4f -- %s", final_test_loss,
                "GROKKED" if grokked else "NOT GROKKED")

    if not grokked:
        row = {
            "operation": operation,
            "p": p,
            "seed": seed,
            "grokked": False,
            "final_test_loss": final_test_loss,
            "winding_number": None,
            "dominant_fourier_mode": None,
            "match": None,
            "iia_k2": None,
            "n_epochs": n_epochs,
            "das_steps": das_steps,
            "timestamp": utc_ts(),
        }
        with open(output_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        logger.info("  Not grokked -- skipping analysis, row appended.")
        return row

    model.eval()
    d = model.cfg.d_model
    layer = 0
    site = site_resid(layer, d)

    # Fit DAS k=2
    cached = cache_pairs(model, dataset, labels, train_idx, layer, device)
    if len(cached) < 10:
        logger.warning("  Too few valid pairs (%d) -- skipping", len(cached))
        return None

    Q, iia = train_das(model, site, cached, 2, das_steps, device)
    logger.info("  DAS k=2: IIA=%.3f", iia)

    # Step 1: Compute per-class centroids in 2D DAS space
    centroids, unique_labels = compute_centroids_2d(
        model, dataset, labels, Q, layer, device,
    )
    logger.info("  Centroids: %d unique labels", len(unique_labels))

    # Ensure centroids are ordered 0..p-1
    n_labels = max(unique_labels) + 1
    ordered_centroids = np.zeros((n_labels, 2))
    label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
    for lbl in unique_labels:
        ordered_centroids[lbl] = centroids[label_to_idx[lbl]]

    # Step 2: Compute winding number
    winding_number, total_angle = compute_winding_number(ordered_centroids)
    logger.info("  Winding number: %d (total angle: %.2f rad = %.2f * 2pi)",
                winding_number, total_angle, total_angle / (2 * math.pi))

    # Step 3: Dominant Fourier mode from embedding weights
    dominant_fourier_mode, fourier_spectrum = compute_dominant_fourier_mode(model, p)
    logger.info("  Dominant Fourier mode: k=%d", dominant_fourier_mode)

    # Step 4: Compare
    # Use absolute value of winding number (direction is arbitrary)
    match = abs(winding_number) == dominant_fourier_mode
    logger.info("  Match: %s (|winding|=%d vs fourier=%d)",
                "YES" if match else "NO",
                abs(winding_number), dominant_fourier_mode)

    # Top 5 Fourier modes for diagnostics
    sorted_spectrum = sorted(fourier_spectrum.items(), key=lambda x: -x[1])[:5]

    row = {
        "operation": operation,
        "p": p,
        "seed": seed,
        "grokked": True,
        "final_test_loss": float(final_test_loss),
        "winding_number": int(winding_number),
        "abs_winding_number": int(abs(winding_number)),
        "total_angle": float(total_angle),
        "dominant_fourier_mode": int(dominant_fourier_mode),
        "match": match,
        "iia_k2": float(iia),
        "n_epochs": n_epochs,
        "das_steps": das_steps,
        "top5_fourier_modes": {str(k): float(v) for k, v in sorted_spectrum},
        "timestamp": utc_ts(),
    }

    with open(output_path, "a") as f:
        f.write(json.dumps(row) + "\n")
    logger.info("  Row appended to %s", output_path)

    return row


# -- Main --

def main():
    parser = argparse.ArgumentParser(
        description="Winding Number Fourier (E8) -- Theorem 2.3: winding == Fourier mode")
    parser.add_argument("--operations",
                        default=",".join(ALWAYS_CLASS_OPS),
                        help="Comma-separated list of operations")
    parser.add_argument("--n-seeds", type=int, default=10,
                        help="Number of seeds per operation")
    parser.add_argument("--p", type=int, default=113,
                        help="Default modulus (overridden per-op where needed)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-epochs", type=int, default=None,
                        help="Override epoch count for all operations")
    parser.add_argument("--das-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42,
                        help="Base seed (actual seed = base_seed + seed_idx)")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results"))
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    operations = [op.strip() for op in args.operations.split(",")]
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
            name=f"winding-fourier-{len(operations)}ops-{args.n_seeds}seeds",
            job_type="geometry_wild",
            tags=["geometry_wild", "winding_number_fourier", "theorem_2_3"],
            config={
                "operations": operations,
                "n_seeds": args.n_seeds,
                "p_default": args.p,
                "n_epochs_override": args.n_epochs,
                "das_steps": args.das_steps,
                "base_seed": args.seed,
            },
        )

    logger.info("[%s] Winding Number Fourier (E8) -- Theorem 2.3", utc_ts())
    logger.info("  Operations: %s", operations)
    logger.info("  Seeds per op: %d, default p: %d, DAS steps: %d, base seed: %d",
                args.n_seeds, args.p, args.das_steps, args.seed)

    output_path = output_dir / "winding_number_fourier.jsonl"

    # Build run list: (operation, p, seed) triples
    run_configs = []
    for op in operations:
        p_val = P_OVERRIDES.get(op, args.p)
        for seed_idx in range(args.n_seeds):
            run_configs.append((op, p_val, args.seed + seed_idx))

    # -- Run all trials --
    results = []
    for op, p_val, seed in tqdm(run_configs, desc="WindingFourier"):
        n_epochs = (args.n_epochs if args.n_epochs is not None
                    else EPOCH_DEFAULTS.get(op, 25000))

        row = run_one_trial(op, p_val, device, n_epochs, args.das_steps,
                            seed, output_path)
        if row is not None:
            results.append(row)
            if wandb_run:
                import wandb
                tag = f"{op}_s{seed}"
                wandb.log({
                    f"per_trial/{tag}/grokked": int(row["grokked"]),
                    f"per_trial/{tag}/winding": row["abs_winding_number"] if row.get("abs_winding_number") is not None else -1,
                    f"per_trial/{tag}/fourier_mode": row["dominant_fourier_mode"] if row["dominant_fourier_mode"] is not None else -1,
                    f"per_trial/{tag}/match": int(row["match"]) if row["match"] is not None else -1,
                    f"per_trial/{tag}/iia_k2": row["iia_k2"] if row["iia_k2"] is not None else 0.0,
                })
        else:
            logger.warning("  Skipped %s seed=%d (fit failed)", op, seed)

    logger.info("\n[%s] === All trials complete: %d / %d succeeded ===",
                utc_ts(), len(results), len(run_configs))

    if not results:
        logger.error("No successful trials -- nothing to summarize.")
        if wandb_run:
            import wandb
            wandb_run.finish(exit_code=1)
        return

    # -- Per-operation aggregation --
    grokked_results = [r for r in results if r["grokked"]]
    matched_results = [r for r in grokked_results if r["match"]]

    overall_match_rate = len(matched_results) / max(len(grokked_results), 1)

    # Per-operation stats
    per_op_stats = {}
    for op in operations:
        op_results = [r for r in results if r["operation"] == op]
        op_grokked = [r for r in op_results if r["grokked"]]
        op_matched = [r for r in op_grokked if r["match"]]

        per_op_stats[op] = {
            "n_total": len(op_results),
            "n_grokked": len(op_grokked),
            "n_matched": len(op_matched),
            "match_rate": len(op_matched) / max(len(op_grokked), 1),
            "winding_numbers": [r["abs_winding_number"] for r in op_grokked],
            "fourier_modes": [r["dominant_fourier_mode"] for r in op_grokked],
            "iia_values": [r["iia_k2"] for r in op_grokked if r["iia_k2"] is not None],
        }

    if wandb_run:
        import wandb
        wandb.log({
            "summary/n_total": len(results),
            "summary/n_grokked": len(grokked_results),
            "summary/n_matched": len(matched_results),
            "summary/overall_match_rate": overall_match_rate,
        })

        for op, stats in per_op_stats.items():
            wandb.log({
                f"per_op/{op}/n_grokked": stats["n_grokked"],
                f"per_op/{op}/n_matched": stats["n_matched"],
                f"per_op/{op}/match_rate": stats["match_rate"],
            })

        results_table = wandb.Table(
            columns=["operation", "seed", "grokked", "winding", "fourier_mode",
                      "match", "iia_k2", "test_loss"],
            data=[[r["operation"], r["seed"], r["grokked"],
                   r.get("abs_winding_number"), r["dominant_fourier_mode"],
                   r["match"], r["iia_k2"], r["final_test_loss"]]
                  for r in results],
        )
        wandb.log({"results_table": results_table})

        artifact = wandb.Artifact("winding-number-fourier-results",
                                  type="geometry_wild_results")
        artifact.add_file(str(output_path))
        wandb_run.log_artifact(artifact)
        wandb_run.finish()

    # -- Summary table --
    print(f"\n{'=' * 100}")
    print(f"Winding Number Fourier (E8) -- Theorem 2.3")
    print(f"Prediction: winding number of DAS centroids == dominant Fourier mode index")
    print(f"{'=' * 100}")

    print(f"\n{'Operation':<22s} {'p':>5s} {'Grok':>5s} {'Match':>6s} "
          f"{'Rate':>6s} {'Wind':>5s} {'Four':>5s}")
    print("-" * 62)
    for op in operations:
        stats = per_op_stats[op]
        wind_str = ",".join(str(w) for w in stats["winding_numbers"][:5])
        four_str = ",".join(str(f) for f in stats["fourier_modes"][:5])
        if len(stats["winding_numbers"]) > 5:
            wind_str += "..."
            four_str += "..."
        p_val = P_OVERRIDES.get(op, args.p)
        print(f"{op:<22s} {p_val:5d} "
              f"{stats['n_grokked']:3d}/{stats['n_total']:<2d} "
              f"{stats['n_matched']:3d}/{stats['n_grokked']:<2d} "
              f"{stats['match_rate']:6.3f} "
              f"{wind_str:>5s} {four_str:>5s}")

    print(f"\nDetailed trial results:")
    print(f"{'Operation':<22s} {'Seed':>5s} {'Grok':>5s} {'|Wind|':>6s} {'Four':>5s} "
          f"{'Match':>6s} {'IIA':>6s} {'TestL':>7s}")
    print("-" * 72)
    for r in results:
        grok_str = "YES" if r["grokked"] else "NO"
        wind_str = str(r.get("abs_winding_number", "")) if r.get("abs_winding_number") is not None else "N/A"
        four_str = str(r["dominant_fourier_mode"]) if r["dominant_fourier_mode"] is not None else "N/A"
        match_str = "YES" if r["match"] else ("NO" if r["match"] is not None else "N/A")
        iia_str = f"{r['iia_k2']:.3f}" if r["iia_k2"] is not None else "N/A"
        print(f"{r['operation']:<22s} {r['seed']:5d} {grok_str:>5s} "
              f"{wind_str:>6s} {four_str:>5s} {match_str:>6s} "
              f"{iia_str:>6s} {r['final_test_loss']:7.4f}")

    # Overall verdict
    print(f"\n{'=' * 100}")
    print(f"Overall match rate (grokked only): {len(matched_results)}/{len(grokked_results)} "
          f"= {100 * overall_match_rate:.1f}%")
    print(f"Prediction: >99% match")
    if overall_match_rate > 0.99:
        print(f"VERDICT: CONFIRMED -- winding number matches Fourier mode in >{100*overall_match_rate:.0f}% of cases")
    elif overall_match_rate > 0.90:
        print(f"VERDICT: MOSTLY CONFIRMED -- {100*overall_match_rate:.1f}% match rate (some deviations)")
    else:
        print(f"VERDICT: NOT CONFIRMED -- only {100*overall_match_rate:.1f}% match rate")

    # Show mismatches if any
    mismatches = [r for r in grokked_results if not r["match"]]
    if mismatches:
        print(f"\nMismatches ({len(mismatches)}):")
        for r in mismatches:
            print(f"  {r['operation']} seed={r['seed']}: "
                  f"|winding|={r['abs_winding_number']}, fourier={r['dominant_fourier_mode']}, "
                  f"IIA={r['iia_k2']:.3f}")

    print(f"\n{'=' * 100}")
    logger.info("[%s] Done.", utc_ts())


if __name__ == "__main__":
    main()
