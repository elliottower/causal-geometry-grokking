"""Fourier Superposition MDL — Open Question 4 from the Minimal Representation Proof.

"Multi-mode Fourier superposition needs a separate MDL-flavored argument."

For each operation, trains a grokking model, fits DAS at multiple k values,
and analyzes the Fourier decomposition of the learned representation:

  1. Train grokking model, fit DAS at k in {2, 4, 6, 8, 12, 16, 24, 32}
  2. For each k, project activations H @ Q and compute the Fourier power spectrum
  3. Count "active" Fourier modes: modes with power > 5% of maximum mode's power
  4. Compute Fourier entropy: H = -sum(p_k log p_k) where p_k = normalized power
  5. Compute MDL score: k_MDL = argmin_k (reconstruction_error + k * log(p) / p)
  6. Group-action ops: should have exactly 1 active mode at k=2 (single irrep)
  7. Polynomial/never-class ops: should have many active modes, no single k suffices

The MDL argument: k* is high for never-class because the operation requires
MANY Fourier modes, each needing 2 real dims.  So k* ~ 2 * n_active_modes.

Usage:
    # Full run (GPU)
    python -u experiments/batch6_atlas/geometry_wild/fourier_superposition_mdl.py \\
        --operations multiplication,subtraction,division,polynomial,cubing,squaring,abs_diff,affine,power,composite_addition \\
        --device cuda --output-dir /workspace/results

    # Local test
    python -u experiments/batch6_atlas/geometry_wild/fourier_superposition_mdl.py \\
        --operations multiplication,subtraction --p 17 --n-epochs 100 \\
        --das-steps 5 --device cpu --output-dir experiments/results
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
import torch.nn.functional as F
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
    get_all_activations,
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

P_OVERRIDES = {
    "composite_addition": 91,
}

# Three-class partition mirroring matroid_stratification / rg_flow_k_sweep.
GROKKING_CLASSES = {
    "always": [
        "multiplication", "subtraction", "division", "bitwise_xor",
        "cubic_sum", "sum_of_squares", "max_ab",
    ],
    "stochastic": ["composite_addition", "power"],
    "never": ["cubing", "squaring", "abs_diff", "polynomial", "affine"],
}

# DAS k values to sweep.
K_SWEEP_VALUES = [2, 4, 6, 8, 12, 16, 24, 32]

# Fourier mode threshold: fraction of max-mode power to be considered "active".
ACTIVE_MODE_THRESHOLD = 0.05


def utc_ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _algebraic_class(operation):
    """Return 'always', 'stochastic', or 'never' for a given operation."""
    for cls, ops in GROKKING_CLASSES.items():
        if operation in ops:
            return cls
    return "unknown"


# ── Fourier power spectrum ──

@torch.no_grad()
def fourier_power_spectrum(model, dataset, labels, Q, layer, p, device):
    """Project activations through DAS subspace, compute Fourier power per mode.

    For each label c in 0..p-1 we compute the centroid of H @ Q, giving a
    (p, k) matrix of centroids.  Then we DFT each column across labels and
    sum squared magnitudes across columns to get per-mode power.

    Returns:
        spectrum: list of dicts {mode_k, power, normalized_power}
        n_active: number of modes with normalized_power > ACTIVE_MODE_THRESHOLD
        entropy: -sum(p_k log p_k) over normalized power distribution
    """
    hook_name = f"blocks.{layer}.hook_resid_post"
    batch_size = 256
    k = Q.shape[1]

    # Collect projected activations per label.
    sums = torch.zeros(p, k, device=device)
    counts = torch.zeros(p, device=device)

    for start in range(0, len(dataset), batch_size):
        batch = dataset[start:start + batch_size]
        batch_labels = labels[start:start + batch_size]
        _, cache = model.run_with_cache(batch, names_filter=[hook_name])
        H = cache[hook_name][:, -1, :]  # (batch, d_model)
        Z = H @ Q  # (batch, k)
        for i in range(Z.shape[0]):
            lbl = batch_labels[i].item()
            if 0 <= lbl < p:
                sums[lbl] += Z[i]
                counts[lbl] += 1

    active_mask = counts > 0
    centroids = torch.zeros(p, k, device=device)
    centroids[active_mask] = sums[active_mask] / counts[active_mask].unsqueeze(1)

    # DFT across the label axis for each of the k columns.
    centroids_np = centroids.cpu().numpy()  # (p, k)
    centroids_np = centroids_np - centroids_np.mean(axis=0, keepdims=True)

    # Power per Fourier mode: sum over k columns of |DFT|^2.
    total_power = np.zeros(p)
    for col in range(k):
        fft_col = np.fft.fft(centroids_np[:, col])
        total_power += np.abs(fft_col) ** 2

    # Only consider positive frequencies up to Nyquist.
    max_freq = p // 2
    mode_powers = {}
    for freq in range(1, max_freq + 1):
        mode_powers[freq] = float(total_power[freq])

    if not mode_powers:
        return [], 0, 0.0

    max_power = max(mode_powers.values())
    if max_power < 1e-15:
        max_power = 1.0

    spectrum = []
    for freq in sorted(mode_powers.keys()):
        pw = mode_powers[freq]
        spectrum.append({
            "mode_k": freq,
            "power": pw,
            "normalized_power": pw / max_power,
        })

    # Count active modes.
    n_active = sum(1 for s in spectrum if s["normalized_power"] > ACTIVE_MODE_THRESHOLD)

    # Fourier entropy over normalized power distribution.
    powers = np.array([s["power"] for s in spectrum])
    total = powers.sum()
    if total > 1e-15:
        probs = powers / total
        probs = probs[probs > 1e-15]
        entropy = -float(np.sum(probs * np.log(probs)))
    else:
        entropy = 0.0

    return spectrum, n_active, entropy


# ── Reconstruction error ──

@torch.no_grad()
def reconstruction_error(model, dataset, labels, Q, layer, p, device):
    """Cross-entropy of the model when restricted to the k-dim DAS subspace.

    Projects activations into Q's column space and measures how well the model
    can predict labels using only that subspace.
    """
    hook_name = f"blocks.{layer}.hook_resid_post"
    proj = Q @ Q.T  # (d_model, d_model)
    batch_size = 256
    total_loss = 0.0
    total_count = 0

    for start in range(0, len(dataset), batch_size):
        batch = dataset[start:start + batch_size]
        batch_labels = labels[start:start + batch_size]
        _, cache = model.run_with_cache(batch, names_filter=[hook_name])
        H = cache[hook_name][:, -1, :]  # (batch, d_model)
        # Replace activations with their projection onto Q's subspace.
        H_proj = H @ proj
        complement = H - H_proj
        # Zero out the complement (keep only DAS subspace).
        # Compute logits using the projected activations via the unembedding.
        logits = H_proj @ model.W_U + model.b_U  # (batch, d_vocab_out)
        loss = F.cross_entropy(logits, batch_labels, reduction="sum")
        total_loss += loss.item()
        total_count += len(batch_labels)

    return total_loss / max(total_count, 1)


# ── MDL score ──

def mdl_score(recon_error, k, p):
    """MDL tradeoff: reconstruction_error + k * log(p) / p.

    The model complexity term k * log(p) / p penalizes large subspace
    dimensions.  For group-action ops needing one Fourier mode (2 real dims),
    k=2 wins.  For multi-mode ops, more modes means larger k is needed to
    reduce reconstruction error, so k_MDL grows.
    """
    model_penalty = k * math.log(p) / p
    return recon_error + model_penalty


# ── Per-operation pipeline ──

def process_operation(operation, p, device, n_epochs, das_steps, seed,
                      k_values, output_path):
    """Train model, sweep k, compute Fourier spectra and MDL scores.

    Appends one JSONL row and returns result dict.
    """
    logger.info("[%s] === %s (p=%d, epochs=%d) ===", utc_ts(), operation, p, n_epochs)

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

    # Cache pairs for DAS training.
    cached = cache_pairs(model, dataset, labels, train_idx, layer, device)
    if len(cached) < 10:
        logger.warning("  Too few valid pairs (%d) — skipping %s", len(cached), operation)
        return None

    # ── k-sweep: DAS + Fourier spectrum + reconstruction + MDL ──
    k_sweep_iia = {}
    mdl_scores = {}
    all_spectra = {}
    all_n_active = {}
    all_entropy = {}
    all_recon_error = {}

    for k in tqdm(k_values, desc=f"  k-sweep ({operation})", leave=False):
        Q, iia = train_das(model, site, cached, k, das_steps, device)
        k_sweep_iia[k] = float(iia)

        spectrum, n_active, entropy = fourier_power_spectrum(
            model, dataset, labels, Q, layer, p, device,
        )
        all_spectra[k] = spectrum
        all_n_active[k] = n_active
        all_entropy[k] = entropy

        recon = reconstruction_error(model, dataset, labels, Q, layer, p, device)
        all_recon_error[k] = recon

        mdl_val = mdl_score(recon, k, p)
        mdl_scores[k] = mdl_val

        logger.info("    k=%2d: IIA=%.3f  n_active=%2d  entropy=%.3f  "
                     "recon=%.4f  MDL=%.4f",
                     k, iia, n_active, entropy, recon, mdl_val)

    # DAS k* = first k with IIA > 0.9 (or argmax if none).
    k_star_das = None
    for k in k_values:
        if k_sweep_iia[k] > 0.9:
            k_star_das = k
            break
    if k_star_das is None:
        k_star_das = max(k_sweep_iia, key=k_sweep_iia.get)

    # MDL optimal k.
    k_mdl = min(mdl_scores, key=mdl_scores.get)

    # Use k=2 and k* results for primary reporting.
    recon_k2 = all_recon_error.get(2, float("nan"))
    recon_kstar = all_recon_error.get(k_star_das, float("nan"))
    iia_k2 = k_sweep_iia.get(2, float("nan"))

    # Fourier spectrum at k* for the mode_spectrum output.
    mode_spectrum = all_spectra.get(k_star_das, all_spectra.get(2, []))
    n_active_modes = all_n_active.get(k_star_das, all_n_active.get(2, 0))
    fourier_entropy = all_entropy.get(k_star_das, all_entropy.get(2, 0.0))

    # Equivariance at k*.
    Q_star, _ = train_das(model, site, cached, k_star_das, das_steps, device)
    equiv = equivariance_test(model, site, Q_star, dataset, labels, device, p, operation)
    logger.info("  Equivariance at k*=%d: %.3f", k_star_das,
                equiv["equivariant_fraction"])

    alg_class = _algebraic_class(operation)

    row = {
        "operation": operation,
        "grokked": grokked,
        "n_active_modes": n_active_modes,
        "fourier_entropy": fourier_entropy,
        "k_mdl": k_mdl,
        "k_star_das": k_star_das,
        "mode_spectrum": mode_spectrum,
        "k_sweep_iia": {str(k): v for k, v in k_sweep_iia.items()},
        "algebraic_class": alg_class,
        "reconstruction_error_k2": recon_k2,
        "reconstruction_error_kstar": recon_kstar,
        "iia_k2": iia_k2,
        "equivariance": float(equiv["equivariant_fraction"]),
        "final_test_loss": float(final_test_loss),
        "timestamp": utc_ts(),
    }

    with open(output_path, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")
    logger.info("  Row appended to %s", output_path)

    return row


# ── Main ──

def main():
    parser = argparse.ArgumentParser(
        description="Fourier Superposition MDL — Open Question 4 "
                    "(multi-mode Fourier superposition)")
    parser.add_argument("--operations", required=True,
                        help="Comma-separated list of operations")
    parser.add_argument("--p", type=int, default=113,
                        help="Default modulus (overridden per-op where needed)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-epochs", type=int, default=None,
                        help="Override epoch count for all operations")
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
        import wandb
        wandb_run = wandb.init(
            project="SAELensCircuitPort - Experimental",
            entity="factorized-circuits",
            name=f"fourier-superposition-mdl-{len(operations)}ops",
            job_type="geometry_wild",
            tags=["geometry_wild", "fourier_superposition_mdl", "open_question_4"],
            config={
                "operations": operations,
                "p_default": args.p,
                "n_epochs_override": args.n_epochs,
                "das_steps": args.das_steps,
                "k_sweep_values": K_SWEEP_VALUES,
                "active_mode_threshold": ACTIVE_MODE_THRESHOLD,
                "seed": args.seed,
            },
        )

    logger.info("[%s] Fourier Superposition MDL — %d operations",
                utc_ts(), len(operations))
    logger.info("  Operations: %s", operations)
    logger.info("  Default p: %d, DAS steps: %d, k values: %s, seed: %d",
                args.p, args.das_steps, K_SWEEP_VALUES, args.seed)

    output_path = output_dir / "fourier_superposition_mdl.jsonl"

    # ── Process each operation ──
    results = []
    for op in tqdm(operations, desc="Fourier MDL"):
        p_op = P_OVERRIDES.get(op, args.p)
        n_epochs_op = (args.n_epochs if args.n_epochs is not None
                       else EPOCH_DEFAULTS.get(op, 25000))

        row = process_operation(
            op, p_op, device, n_epochs_op, args.das_steps, args.seed,
            K_SWEEP_VALUES, output_path,
        )
        if row is not None:
            results.append(row)
            if wandb_run:
                import wandb
                wandb.log({
                    f"per_op/{op}/n_active_modes": row["n_active_modes"],
                    f"per_op/{op}/fourier_entropy": row["fourier_entropy"],
                    f"per_op/{op}/k_mdl": row["k_mdl"],
                    f"per_op/{op}/k_star_das": row["k_star_das"],
                    f"per_op/{op}/iia_k2": row["iia_k2"],
                    f"per_op/{op}/equivariance": row["equivariance"],
                    f"per_op/{op}/grokked": int(row["grokked"]),
                })
        else:
            logger.warning("  Skipped %s (processing failed)", op)

    logger.info("\n[%s] === All operations complete: %d / %d succeeded ===",
                utc_ts(), len(results), len(operations))

    if not results:
        logger.error("No operations completed successfully.")
        if wandb_run:
            import wandb
            wandb_run.finish(exit_code=1)
        return

    # ── Per-class analysis ──
    logger.info("\n[%s] === Per-class summary ===", utc_ts())
    for cls_name in ("always", "stochastic", "never"):
        cls_results = [r for r in results if r["algebraic_class"] == cls_name]
        if not cls_results:
            continue
        active_modes = [r["n_active_modes"] for r in cls_results]
        entropies = [r["fourier_entropy"] for r in cls_results]
        k_mdls = [r["k_mdl"] for r in cls_results]
        k_stars = [r["k_star_das"] for r in cls_results]
        logger.info("  %s (%d ops): active_modes=%.1f+/-%.1f  "
                    "entropy=%.2f+/-%.2f  k_mdl=%.1f  k*=%.1f",
                    cls_name, len(cls_results),
                    np.mean(active_modes), np.std(active_modes),
                    np.mean(entropies), np.std(entropies),
                    np.mean(k_mdls), np.mean(k_stars))

        if wandb_run:
            import wandb
            wandb.log({
                f"class/{cls_name}/mean_active_modes": float(np.mean(active_modes)),
                f"class/{cls_name}/mean_entropy": float(np.mean(entropies)),
                f"class/{cls_name}/mean_k_mdl": float(np.mean(k_mdls)),
                f"class/{cls_name}/mean_k_star": float(np.mean(k_stars)),
            })

    # ── Correlation: n_active_modes vs k* ──
    grokked_results = [r for r in results if r["grokked"]]
    if len(grokked_results) >= 3:
        from scipy.stats import spearmanr
        active = [r["n_active_modes"] for r in grokked_results]
        k_stars = [r["k_star_das"] for r in grokked_results]
        rho, pval = spearmanr(active, k_stars)
        logger.info("  Spearman(n_active_modes, k*): rho=%.3f, p=%.4f", rho, pval)

        if wandb_run:
            import wandb
            wandb.log({
                "correlations/active_modes_vs_kstar/rho": float(rho),
                "correlations/active_modes_vs_kstar/p": float(pval),
            })

    # ── W&B artifacts ──
    if wandb_run:
        import wandb
        results_table = wandb.Table(
            columns=["operation", "class", "grokked", "n_active", "entropy",
                     "k_mdl", "k_star", "iia_k2", "equiv", "loss"],
            data=[[r["operation"], r["algebraic_class"], r["grokked"],
                   r["n_active_modes"], round(r["fourier_entropy"], 3),
                   r["k_mdl"], r["k_star_das"], round(r["iia_k2"], 3),
                   round(r["equivariance"], 3), round(r["final_test_loss"], 4)]
                  for r in results],
        )
        wandb.log({"results_table": results_table})

        artifact = wandb.Artifact("fourier-superposition-mdl",
                                  type="geometry_wild_results")
        artifact.add_file(str(output_path))
        wandb_run.log_artifact(artifact)
        wandb_run.finish()

    # ── Summary table ──
    print(f"\n{'=' * 105}")
    print(f"Fourier Superposition MDL — {len(results)} operations")
    print(f"{'=' * 105}")

    print(f"\n{'Operation':<22s} {'Class':<12s} {'Grok':>5s} {'Active':>7s} "
          f"{'Entropy':>8s} {'k_MDL':>5s} {'k*':>4s} {'IIA@2':>6s} "
          f"{'Equiv':>6s} {'Loss':>8s}")
    print("-" * 95)
    for r in results:
        print(f"{r['operation']:<22s} {r['algebraic_class']:<12s} "
              f"{'YES' if r['grokked'] else 'NO':>5s} "
              f"{r['n_active_modes']:7d} "
              f"{r['fourier_entropy']:8.3f} "
              f"{r['k_mdl']:5d} "
              f"{r['k_star_das']:4d} "
              f"{r['iia_k2']:6.3f} "
              f"{r['equivariance']:6.3f} "
              f"{r['final_test_loss']:8.4f}")

    # ── Prediction checks ──
    always_results = [r for r in results
                      if r["algebraic_class"] == "always" and r["grokked"]]
    never_results = [r for r in results
                     if r["algebraic_class"] == "never" and r["grokked"]]

    print(f"\nPrediction checks (Open Question 4):")

    if always_results:
        always_active = [r["n_active_modes"] for r in always_results]
        print(f"  Always-class active modes: {always_active}")
        mean_always = np.mean(always_active)
        print(f"    Mean={mean_always:.1f} (prediction: ~1 active mode for group-action ops)")

    if never_results:
        never_active = [r["n_active_modes"] for r in never_results]
        print(f"  Never-class active modes: {never_active}")
        mean_never = np.mean(never_active)
        print(f"    Mean={mean_never:.1f} (prediction: many active modes for non-group ops)")

    if always_results and never_results:
        sep = np.mean([r["n_active_modes"] for r in never_results]) - \
              np.mean([r["n_active_modes"] for r in always_results])
        print(f"  Separation (never - always): {sep:.1f} "
              f"({'SUPPORTED' if sep > 2 else 'WEAK' if sep > 0 else 'FALSIFIED'})")

    if len(grokked_results) >= 3:
        active = [r["n_active_modes"] for r in grokked_results]
        k_stars = [r["k_star_das"] for r in grokked_results]
        rho, _ = spearmanr(active, k_stars)
        print(f"  n_active_modes correlates with k*: rho={rho:.3f} "
              f"({'SUPPORTED' if rho > 0.3 else 'WEAK' if rho > 0 else 'FALSIFIED'})")

    # MDL argument: k_MDL ~ 2 * n_active for grokked ops
    if grokked_results:
        print(f"\n  MDL argument check (k_MDL ~ 2 * n_active_modes):")
        for r in grokked_results:
            ratio = r["k_mdl"] / max(r["n_active_modes"], 1)
            print(f"    {r['operation']:<22s}: k_MDL={r['k_mdl']:3d}  "
                  f"n_active={r['n_active_modes']:3d}  ratio={ratio:.1f}")

    print(f"\n{'=' * 105}")
    logger.info("[%s] Done.", utc_ts())


if __name__ == "__main__":
    main()
