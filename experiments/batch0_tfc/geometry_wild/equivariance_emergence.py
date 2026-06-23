"""Equivariance emergence — Open Question 1 from minimal representation proof.

Is the representation-theoretic structure an attractor of training dynamics?

For each operation, train with dense checkpoints and at each checkpoint:
  1. Fit DAS at k=2
  2. Project activations H @ Q to get 2D points
  3. Compute per-label centroids c_g for each group element g
  4. For each pair (g, g'), compute the empirical "action matrix"
     A_{g->g'} = argmin_A ||A c_g - c_{g'}||
  5. Measure how close A is to a rotation matrix:
     rotation_error = ||A - nearest_rotation(A)||_F
  6. Compute equivariance score via equivariance_test
  7. Track the dominant Fourier mode k at each checkpoint

Key output: a trajectory showing rotation_error vs epoch (should drop at
grokking), equivariance vs epoch (should jump at grokking), and whether the
rotation error drop PRECEDES or FOLLOWS the loss transition. The final
rotation angle is compared to the theoretical 2*pi*k/p.

Usage:
    # Full run (group-action ops, GPU)
    python -u experiments/batch6_atlas/geometry_wild/equivariance_emergence.py \\
        --operations multiplication,subtraction,division \\
        --n-epochs 40000 --ckpt-interval 200 --das-steps 400 \\
        --device cuda --output-dir /workspace/results

    # Local test
    python -u experiments/batch6_atlas/geometry_wild/equivariance_emergence.py \\
        --operations multiplication --p 17 --n-epochs 500 \\
        --ckpt-interval 100 --das-steps 5 --device cpu --no-wandb
"""
from __future__ import annotations

import argparse
import copy
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
    equivariance_test,
    train_das,
    train_grokking_model,
)

from factorization_circuits.pipeline.utils.factor_das_kernel import site_resid

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

EPOCH_DEFAULTS = {
    "multiplication": 40000, "polynomial": 60000,
    "composite_addition": 15000, "division": 40000,
    "squaring": 60000, "cubing": 60000,
    "max_ab": 25000, "abs_diff": 60000, "sum_of_squares": 30000,
    "power": 80000, "shifted_mult": 40000,
    "min_ab": 60000, "floor_div": 60000, "bitwise_xor": 60000,
    "gcd": 60000, "subtraction": 60000, "affine": 60000,
    "cubic_sum": 60000, "modular_distance": 60000,
    "quartic_sum": 60000, "quintic_sum": 60000, "affine_scaled": 60000,
}

P_DEFAULTS = {
    "composite_addition": 91,
}

P_OVERRIDES = {
    "composite_addition": 91,
}


def utc_ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Nearest rotation and action matrix fitting ──

def nearest_rotation(A):
    """Find the nearest rotation matrix to A using SVD (Procrustes).

    For a 2x2 matrix A, the nearest rotation R = U @ V^T where A = U S V^T,
    with det(R) forced to +1.
    """
    U, _, Vt = np.linalg.svd(A)
    R = U @ Vt
    # Ensure det(R) = +1 (proper rotation, not reflection)
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def rotation_angle(R):
    """Extract the rotation angle from a 2x2 rotation matrix.

    angle = atan2(R[1,0], R[0,0]), returned in [0, 2*pi).
    """
    angle = math.atan2(R[1, 0], R[0, 0])
    if angle < 0:
        angle += 2 * math.pi
    return angle


def fit_action_matrix(c_source, c_target):
    """Fit 2x2 matrix A such that A @ c_source ~ c_target via least squares.

    c_source: (2,) centroid of source label
    c_target: (2,) centroid of target label
    Returns A as (2, 2) array.
    """
    # A @ c_source = c_target  =>  solve for A given one pair
    # With a single pair, this is underdetermined. We use the
    # pseudoinverse: A = c_target @ c_source^+ where c_source is (2,).
    # For a single vector, c^+ = c / ||c||^2.
    norm_sq = np.dot(c_source, c_source)
    if norm_sq < 1e-12:
        return np.eye(2)
    # A = outer(c_target, c_source) / ||c_source||^2
    A = np.outer(c_target, c_source) / norm_sq
    return A


def fit_global_action_matrix(centroids, label_pairs):
    """Fit a single 2x2 matrix A that best maps c_g -> c_{g'} for all pairs.

    centroids: dict mapping label -> (2,) array
    label_pairs: list of (source_label, target_label)

    Solves: min_A sum_{(g,g')} ||A @ c_g - c_{g'}||^2 via stacking.
    """
    sources = []
    targets = []
    for g, gp in label_pairs:
        if g in centroids and gp in centroids:
            sources.append(centroids[g])
            targets.append(centroids[gp])

    if len(sources) < 2:
        return np.eye(2), float("inf")

    S = np.array(sources)  # (n, 2)
    T = np.array(targets)  # (n, 2)

    # Solve T = S @ A^T  =>  A^T = pinv(S) @ T  =>  A = (pinv(S) @ T)^T
    A_T, residuals, _, _ = np.linalg.lstsq(S, T, rcond=None)
    A = A_T.T

    # Compute residual
    T_pred = S @ A.T
    residual = np.sqrt(np.mean(np.sum((T - T_pred) ** 2, axis=1)))
    return A, residual


# ── Centroid computation ──

@torch.no_grad()
def compute_centroids_dict(model, dataset, labels, Q, layer, device):
    """Forward pass, project H @ Q into 2D, return dict: label -> centroid (2,)."""
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

    centroids = {}
    for lbl in sorted(set(label_arr.tolist())):
        mask = label_arr == lbl
        centroids[int(lbl)] = projected[mask].mean(axis=0)

    return centroids


# ── Fourier analysis ──

def dominant_fourier_mode(centroids_dict, p):
    """Find the dominant Fourier frequency in the centroid arrangement.

    Orders centroids by label 0..p-1, treats (x, y) as complex z = x + iy,
    computes DFT and returns the frequency with the largest magnitude.

    Returns (dominant_freq, spectrum_dict).
    """
    ordered = np.zeros((p, 2))
    for lbl in range(p):
        if lbl in centroids_dict:
            ordered[lbl] = centroids_dict[lbl]

    z = ordered[:, 0] + 1j * ordered[:, 1]
    z = z - z.mean()

    fft_vals = np.fft.fft(z)
    magnitudes = np.abs(fft_vals)

    max_freq = p // 2
    spectrum = {}
    for freq in range(1, max_freq + 1):
        spectrum[freq] = float(magnitudes[freq])

    if spectrum:
        dominant_freq = max(spectrum, key=spectrum.get)
    else:
        dominant_freq = 0

    return dominant_freq, spectrum


# ── Rotation-error and equivariance at one checkpoint ──

def analyze_checkpoint(model, site, Q, dataset, labels, layer, device, p,
                       operation):
    """At a single checkpoint with fitted DAS Q, compute rotation metrics.

    Returns dict with rotation_error, rotation_angle, equivariance,
    dominant_fourier_k, iia (the Q was already fitted, iia passed separately).
    """
    centroids = compute_centroids_dict(model, dataset, labels, Q, layer, device)

    if len(centroids) < 3:
        return {
            "rotation_error": float("nan"),
            "rotation_angle": float("nan"),
            "equivariance": 0.0,
            "dominant_fourier_k": 0,
        }

    # Compute the empirical action matrix for the shift g -> g+1 (additive)
    # or g -> 2*g (multiplicative), using all available pairs.
    multiplicative_ops = ("multiplication", "division", "shifted_mult", "power")
    if operation in multiplicative_ops:
        # Multiplicative shift: g -> 2*g mod p
        shift_fn = lambda g: (2 * g) % p
    else:
        # Additive shift: g -> g+1 mod p
        shift_fn = lambda g: (g + 1) % p

    label_pairs = []
    for g in centroids:
        gp = shift_fn(g)
        if gp in centroids:
            label_pairs.append((g, gp))

    if len(label_pairs) < 3:
        return {
            "rotation_error": float("nan"),
            "rotation_angle": float("nan"),
            "equivariance": 0.0,
            "dominant_fourier_k": 0,
        }

    A, fit_residual = fit_global_action_matrix(centroids, label_pairs)
    R = nearest_rotation(A)
    rot_error = float(np.linalg.norm(A - R, 'fro'))
    rot_angle = rotation_angle(R)

    # Equivariance test
    equiv = equivariance_test(model, site, Q, dataset, labels, device, p,
                              operation=operation, n_test=200)

    # Dominant Fourier mode
    dom_freq, _ = dominant_fourier_mode(centroids, p)

    return {
        "rotation_error": rot_error,
        "rotation_angle": rot_angle,
        "equivariance": equiv["equivariant_fraction"],
        "dominant_fourier_k": dom_freq,
    }


# ── Convergence epoch detection ──

def find_convergence_epoch(epochs, values, threshold, direction="above"):
    """Find the first epoch where values stay above/below threshold.

    direction="above": first epoch where value >= threshold and stays there.
    direction="below": first epoch where value <= threshold and stays there.
    """
    if len(epochs) < 3:
        return None

    for i in range(len(values)):
        if direction == "above" and values[i] >= threshold:
            # Check it stays above for the rest (or at least 80% of remaining)
            remaining = values[i:]
            if len(remaining) <= 1 or sum(v >= threshold * 0.9 for v in remaining) / len(remaining) > 0.7:
                return epochs[i]
        elif direction == "below" and values[i] <= threshold:
            remaining = values[i:]
            if len(remaining) <= 1 or sum(v <= threshold * 1.1 for v in remaining) / len(remaining) > 0.7:
                return epochs[i]
    return None


def find_loss_transition_epoch(epochs, test_losses):
    """Find the epoch where test loss drops below 0.5 (midpoint of grokking)."""
    for i, tl in enumerate(test_losses):
        if tl is not None and tl < 0.5:
            return epochs[i]
    return None


# ── Per-operation pipeline ──

def run_one_operation(operation, p, device, n_epochs, ckpt_interval, das_steps,
                      seed, output_path):
    """Train with checkpoints, fit DAS at each, track equivariance emergence.

    Appends one JSONL row to output_path and returns the result dict.
    """
    logger.info("[%s] === %s (p=%d) ===", utc_ts(), operation, p)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dataset, labels, train_idx, test_idx = build_data(operation, p, device,
                                                       seed=seed)
    logger.info("  Dataset: %d examples (%d train, %d test)",
                len(dataset), len(train_idx), len(test_idx))

    # Train with checkpointing
    logger.info("[%s] Training grokking model (%d epochs, ckpt every %d)...",
                utc_ts(), n_epochs, ckpt_interval)
    model, cfg, training_data = train_grokking_model(
        p, device, n_epochs=n_epochs, checkpoint_every=ckpt_interval,
        dataset=dataset, labels=labels,
        train_idx=train_idx, test_idx=test_idx, seed=seed,
    )
    checkpoints = training_data["checkpoints"]
    checkpoint_epochs = training_data["checkpoint_epochs"]
    train_losses = training_data["train_losses"]
    test_losses = training_data["test_losses"]

    final_test_loss = test_losses[-1] if test_losses else float("inf")
    grokked = final_test_loss < 0.1
    logger.info("  Final test loss: %.4f -- %s",
                final_test_loss, "GROKKED" if grokked else "NOT GROKKED")

    n_ckpts = len(checkpoints)
    if n_ckpts == 0:
        logger.warning("  No checkpoints saved -- skipping %s", operation)
        return None

    model.eval()
    d = cfg.d_model
    layer = 0
    k = 2
    site = site_resid(layer, d)

    # ── Analyze each checkpoint ──
    trajectory = []
    logger.info("[%s] Analyzing %d checkpoints for equivariance emergence...",
                utc_ts(), n_ckpts)

    for i in tqdm(range(n_ckpts), desc=f"checkpoints ({operation})"):
        epoch = checkpoint_epochs[i]
        model.load_state_dict(checkpoints[i])
        model.eval()

        # Fit DAS
        cached = cache_pairs(model, dataset, labels, train_idx, layer, device,
                             n_pairs=200)
        if len(cached) < 10:
            tl = test_losses[epoch] if epoch < len(test_losses) else None
            trl = train_losses[epoch] if epoch < len(train_losses) else None
            trajectory.append({
                "epoch": epoch,
                "train_loss": trl,
                "test_loss": tl,
                "equivariance": 0.0,
                "rotation_error": float("nan"),
                "dominant_fourier_k": 0,
                "rotation_angle": float("nan"),
                "iia": 0.0,
            })
            continue

        Q, iia = train_das(model, site, cached, k, das_steps, device)

        # Analyze rotation/equivariance at this checkpoint
        metrics = analyze_checkpoint(model, site, Q, dataset, labels, layer,
                                     device, p, operation)

        tl = test_losses[epoch] if epoch < len(test_losses) else None
        trl = train_losses[epoch] if epoch < len(train_losses) else None

        step_result = {
            "epoch": epoch,
            "train_loss": trl,
            "test_loss": tl,
            "equivariance": metrics["equivariance"],
            "rotation_error": metrics["rotation_error"],
            "dominant_fourier_k": metrics["dominant_fourier_k"],
            "rotation_angle": metrics["rotation_angle"],
            "iia": float(iia),
        }
        trajectory.append(step_result)

        tl_str = f"{tl:.4f}" if tl is not None else "N/A"
        re_str = f"{metrics['rotation_error']:.4f}" if not math.isnan(metrics['rotation_error']) else "N/A"
        logger.info("  Epoch %6d: equiv=%.3f  rot_err=%s  fourier_k=%d  "
                    "IIA=%.3f  test_loss=%s",
                    epoch, metrics["equivariance"], re_str,
                    metrics["dominant_fourier_k"], iia, tl_str)

    # Free checkpoint memory
    del checkpoints

    # ── Post-hoc analysis: convergence timing ──
    traj_epochs = [t["epoch"] for t in trajectory]
    traj_equiv = [t["equivariance"] for t in trajectory]
    traj_rot_err = [t["rotation_error"] for t in trajectory]
    traj_test_loss = [t["test_loss"] for t in trajectory]

    # Find convergence epochs
    convergence_epoch_equivariance = find_convergence_epoch(
        traj_epochs, traj_equiv, threshold=0.5, direction="above")
    convergence_epoch_rotation = find_convergence_epoch(
        traj_epochs, traj_rot_err, threshold=0.3, direction="below")
    loss_transition_epoch = find_loss_transition_epoch(traj_epochs, traj_test_loss)

    # Determine temporal ordering
    rotation_precedes_loss = None
    if convergence_epoch_rotation is not None and loss_transition_epoch is not None:
        rotation_precedes_loss = convergence_epoch_rotation < loss_transition_epoch

    # Final rotation angle analysis
    final_step = trajectory[-1]
    final_rotation_angle = final_step["rotation_angle"]
    final_dom_k = final_step["dominant_fourier_k"]

    if final_dom_k > 0 and not math.isnan(final_rotation_angle):
        theoretical_angle = 2 * math.pi * final_dom_k / p
        angle_diff = abs(final_rotation_angle - theoretical_angle)
        # Wrap to [0, pi]
        angle_diff = min(angle_diff, 2 * math.pi - angle_diff)
        angle_match = angle_diff < 0.3  # within ~17 degrees
    else:
        theoretical_angle = float("nan")
        angle_match = False

    row = {
        "operation": operation,
        "grokked": grokked,
        "trajectory": trajectory,
        "convergence_epoch_equivariance": convergence_epoch_equivariance,
        "convergence_epoch_rotation": convergence_epoch_rotation,
        "rotation_precedes_loss": rotation_precedes_loss,
        "final_rotation_angle": final_rotation_angle if not math.isnan(final_rotation_angle) else None,
        "theoretical_angle": theoretical_angle if not math.isnan(theoretical_angle) else None,
        "angle_match": angle_match,
        "final_test_loss": float(final_test_loss),
        "p": int(p),
        "n_epochs": n_epochs,
        "ckpt_interval": ckpt_interval,
        "das_steps": das_steps,
        "seed": seed,
        "timestamp": utc_ts(),
    }

    with open(output_path, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")
    logger.info("  Row appended to %s", output_path)

    return row


# ── Main ──

def main():
    parser = argparse.ArgumentParser(
        description="Equivariance emergence — rotation structure as training attractor")
    parser.add_argument("--operations", required=True,
                        help="Comma-separated list of operations")
    parser.add_argument("--p", type=int, default=None,
                        help="Modulus (default: 113, or per-op override)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-epochs", type=int, default=None,
                        help="Override epoch count for all operations")
    parser.add_argument("--ckpt-interval", type=int, default=200,
                        help="Checkpoint every N epochs")
    parser.add_argument("--das-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
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
        try:
            import wandb
            wandb_run = wandb.init(
                project="SAELensCircuitPort - Experimental",
                entity="factorized-circuits",
                name=f"equivariance-emergence-{len(operations)}ops",
                job_type="geometry_wild",
                tags=["geometry_wild", "equivariance_emergence", "open_question_1"],
                config={
                    "operations": operations,
                    "p": args.p,
                    "n_epochs": args.n_epochs,
                    "ckpt_interval": args.ckpt_interval,
                    "das_steps": args.das_steps,
                    "seed": args.seed,
                },
            )
        except Exception as e:
            logger.warning("W&B init failed: %s -- continuing without W&B", e)

    logger.info("[%s] Equivariance Emergence — %d operations", utc_ts(), len(operations))
    logger.info("  Operations: %s", operations)
    logger.info("  ckpt_interval: %d, DAS steps: %d, seed: %d",
                args.ckpt_interval, args.das_steps, args.seed)

    output_path = output_dir / "equivariance_emergence.jsonl"

    results = []
    for op in tqdm(operations, desc="Operations"):
        p = args.p if args.p is not None else P_OVERRIDES.get(op, 113)
        n_epochs = args.n_epochs or EPOCH_DEFAULTS.get(op, 25000)

        row = run_one_operation(
            operation=op, p=p, device=device, n_epochs=n_epochs,
            ckpt_interval=args.ckpt_interval, das_steps=args.das_steps,
            seed=args.seed, output_path=output_path,
        )
        if row is not None:
            results.append(row)
            if wandb_run is not None:
                import wandb
                # Log trajectory as step-wise metrics
                for step in row["trajectory"]:
                    log_dict = {
                        f"{op}/equivariance": step["equivariance"],
                        f"{op}/iia": step["iia"],
                        f"{op}/dominant_fourier_k": step["dominant_fourier_k"],
                    }
                    if not math.isnan(step["rotation_error"]):
                        log_dict[f"{op}/rotation_error"] = step["rotation_error"]
                    if step["test_loss"] is not None:
                        log_dict[f"{op}/test_loss"] = step["test_loss"]
                    if not math.isnan(step["rotation_angle"]):
                        log_dict[f"{op}/rotation_angle"] = step["rotation_angle"]
                    wandb.log(log_dict, step=step["epoch"])

                # Log summary
                wandb.log({
                    f"summary/{op}/grokked": int(row["grokked"]),
                    f"summary/{op}/convergence_epoch_equivariance": row["convergence_epoch_equivariance"],
                    f"summary/{op}/convergence_epoch_rotation": row["convergence_epoch_rotation"],
                    f"summary/{op}/rotation_precedes_loss": row["rotation_precedes_loss"],
                    f"summary/{op}/angle_match": int(row["angle_match"]),
                })
        else:
            logger.warning("  Skipped %s (no checkpoints)", op)

    logger.info("\n[%s] === All operations complete: %d / %d succeeded ===",
                utc_ts(), len(results), len(operations))

    if not results:
        logger.error("No successful runs -- nothing to summarize.")
        if wandb_run is not None:
            import wandb
            wandb_run.finish(exit_code=1)
        return

    # ── Summary table ──
    print(f"\n{'=' * 100}")
    print(f"Equivariance Emergence — {len(results)} operations")
    print(f"{'=' * 100}")

    print(f"\n{'Operation':<22s} {'Grok':>5s} {'ConvEquiv':>10s} {'ConvRot':>10s} "
          f"{'RotPrecedes':>12s} {'FinalAngle':>11s} {'TheoAngle':>10s} "
          f"{'AngleMatch':>11s} {'TestL':>7s}")
    print("-" * 105)
    for r in results:
        ce_str = f"{r['convergence_epoch_equivariance']}" if r["convergence_epoch_equivariance"] is not None else "N/A"
        cr_str = f"{r['convergence_epoch_rotation']}" if r["convergence_epoch_rotation"] is not None else "N/A"
        rp_str = "YES" if r["rotation_precedes_loss"] is True else ("NO" if r["rotation_precedes_loss"] is False else "N/A")
        fa_str = f"{r['final_rotation_angle']:.4f}" if r["final_rotation_angle"] is not None else "N/A"
        ta_str = f"{r['theoretical_angle']:.4f}" if r["theoretical_angle"] is not None else "N/A"
        am_str = "YES" if r["angle_match"] else "no"

        print(f"{r['operation']:<22s} "
              f"{'YES' if r['grokked'] else 'NO':>5s} "
              f"{ce_str:>10s} {cr_str:>10s} {rp_str:>12s} "
              f"{fa_str:>11s} {ta_str:>10s} {am_str:>11s} "
              f"{r['final_test_loss']:7.4f}")

    # ── Diagnosis ──
    grokked_results = [r for r in results if r["grokked"]]

    if grokked_results:
        print(f"\nDiagnosis (grokked operations only):")

        # Rotation precedes loss?
        n_precedes = sum(1 for r in grokked_results if r["rotation_precedes_loss"] is True)
        n_follows = sum(1 for r in grokked_results if r["rotation_precedes_loss"] is False)
        n_unknown = sum(1 for r in grokked_results if r["rotation_precedes_loss"] is None)
        print(f"  Rotation precedes loss: {n_precedes}/{len(grokked_results)} "
              f"(follows: {n_follows}, unknown: {n_unknown})")
        if n_precedes > n_follows:
            print(f"  FINDING: Rotation structure emerges BEFORE generalization "
                  f"-- supports attractor hypothesis")
        elif n_follows > n_precedes:
            print(f"  FINDING: Rotation structure emerges AFTER generalization "
                  f"-- consequence, not cause")

        # Angle matching
        n_angle_match = sum(1 for r in grokked_results if r["angle_match"])
        print(f"  Angle matches 2*pi*k/p: {n_angle_match}/{len(grokked_results)}")
        if n_angle_match > len(grokked_results) / 2:
            print(f"  FINDING: Rotation angles match Fourier prediction "
                  f"-- consistent with representation theory")

    # Sampled trajectory for each operation
    for r in results:
        traj = r["trajectory"]
        if not traj:
            continue
        print(f"\n  {r['operation']} trajectory (sampled):")
        n_show = min(8, len(traj))
        step = max(1, len(traj) // n_show)
        shown = traj[::step]
        if traj[-1] not in shown:
            shown.append(traj[-1])

        print(f"    {'Epoch':>8s} {'Equiv':>7s} {'RotErr':>8s} {'FourierK':>9s} "
              f"{'IIA':>6s} {'TestLoss':>9s}")
        for t in shown:
            tl_str = f"{t['test_loss']:.4f}" if t["test_loss"] is not None else "N/A"
            re_str = f"{t['rotation_error']:.4f}" if not math.isnan(t["rotation_error"]) else "N/A"
            print(f"    {t['epoch']:8d} {t['equivariance']:7.3f} {re_str:>8s} "
                  f"{t['dominant_fourier_k']:9d} {t['iia']:6.3f} {tl_str:>9s}")

    print(f"\n{'=' * 100}")

    if wandb_run is not None:
        import wandb
        artifact = wandb.Artifact("equivariance-emergence-results",
                                  type="geometry_wild_results")
        artifact.add_file(str(output_path))
        wandb_run.log_artifact(artifact)
        wandb_run.finish()

    logger.info("[%s] Done. Results in %s", utc_ts(), output_path)


if __name__ == "__main__":
    main()
