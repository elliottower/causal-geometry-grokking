"""Grokking onset predictors — which signal fires first?

Addresses Open Question 2 from the minimal representation proof:
"No theorem yet for when grokking occurs."

For each operation, train with dense checkpoints and at each checkpoint
track MULTIPLE signals:
  1. Spectral gap: eigenvalues of H^T H / N, gap = lambda_2 - lambda_3
  2. Fourier coefficient magnitude: project activations to Fourier basis,
     measure |c_k| for each k
  3. DAS IIA at k=2: how well does 2D DAS work at this checkpoint
  4. Equivariance score: from equivariance_test
  5. Weight norm ratio: ||W_fourier|| / ||W_total|| (how much weight is in
     Fourier-aligned directions)
  6. Train/test loss ratio: memorization vs generalization

Then analyze:
  - Which signal crosses its threshold FIRST? (earliest predictor)
  - Compute lead time: how many epochs before loss transition does each
    signal fire
  - Rank signals by predictive power

Usage:
    # Full run (GPU)
    python -u experiments/batch6_atlas/geometry_wild/grokking_onset_predictors.py \\
        --operations multiplication,subtraction,division \\
        --device cuda --output-dir /workspace/results

    # Single operation
    python -u experiments/batch6_atlas/geometry_wild/grokking_onset_predictors.py \\
        --operations multiplication --device cuda

    # Local test
    python -u experiments/batch6_atlas/geometry_wild/grokking_onset_predictors.py \\
        --operations multiplication --p 17 --n-epochs 500 \\
        --ckpt-interval 100 --das-steps 5 --device cpu --no-wandb
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

# Signal thresholds — a signal "fires" when it crosses its threshold.
SIGNAL_THRESHOLDS = {
    "spectral_gap": 0.1,         # gap between lambda_2 and lambda_3
    "fourier_magnitude": 0.3,    # max |c_k| in Fourier basis
    "iia_k2": 0.6,              # DAS IIA at k=2
    "equivariance": 0.3,        # equivariance fraction
    "weight_norm_ratio": 0.1,   # ||W_fourier|| / ||W_total||
    "loss_ratio": 2.0,          # train_loss / test_loss (fires when < threshold)
}

SIGNAL_NAMES = list(SIGNAL_THRESHOLDS.keys())


def utc_ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Signal computations ──

@torch.no_grad()
def compute_spectral_gap(model, dataset, layer, device):
    """Compute spectral gap of H^T H / N at the residual stream.

    Returns (gap, eigenvalues_top5) where gap = lambda_2 - lambda_3.
    """
    hook_name = f"blocks.{layer}.hook_resid_post"
    batch_size = 256
    all_acts = []
    for start in range(0, len(dataset), batch_size):
        batch = dataset[start:start + batch_size]
        _, cache = model.run_with_cache(batch, names_filter=[hook_name])
        H = cache[hook_name][:, -1, :]  # (batch, d_model)
        all_acts.append(H.cpu())

    H_all = torch.cat(all_acts, dim=0).float()  # (N, d_model)
    N = H_all.shape[0]

    # Covariance-like matrix H^T H / N
    cov = H_all.T @ H_all / N
    eigenvalues = torch.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues.flip(0)  # descending order

    top5 = eigenvalues[:5].tolist()
    # gap = lambda_2 - lambda_3 (indices 1 and 2 in 0-indexed)
    gap = 0.0
    if len(eigenvalues) >= 3:
        gap = float(eigenvalues[1] - eigenvalues[2])

    return gap, top5


@torch.no_grad()
def compute_fourier_magnitude(model, dataset, labels, layer, device, p):
    """Project activations to Fourier basis, measure max |c_k|.

    For each Fourier frequency k, compute the average activation weighted
    by exp(2pi i k label / p). The magnitude of the largest coefficient
    indicates Fourier structure.

    Returns (max_magnitude, dominant_k).
    """
    hook_name = f"blocks.{layer}.hook_resid_post"
    batch_size = 256
    all_acts = []
    all_labels = []
    for start in range(0, len(dataset), batch_size):
        batch = dataset[start:start + batch_size]
        batch_labels = labels[start:start + batch_size]
        _, cache = model.run_with_cache(batch, names_filter=[hook_name])
        H = cache[hook_name][:, -1, :]
        all_acts.append(H.cpu())
        all_labels.append(batch_labels.cpu())

    H_all = torch.cat(all_acts, dim=0).float()  # (N, d_model)
    labels_all = torch.cat(all_labels, dim=0)
    N = H_all.shape[0]
    d = H_all.shape[1]

    # Compute Fourier coefficients: c_k = (1/N) sum_n H_n * exp(-2pi i k label_n / p)
    max_magnitude = 0.0
    dominant_k = 0
    max_freq = min(p // 2, 20)  # limit search

    for k in range(1, max_freq + 1):
        phases = 2.0 * math.pi * k * labels_all.float() / p  # (N,)
        cos_phases = torch.cos(phases).unsqueeze(1)  # (N, 1)
        sin_phases = torch.sin(phases).unsqueeze(1)
        c_real = (H_all * cos_phases).mean(dim=0)  # (d_model,)
        c_imag = (H_all * sin_phases).mean(dim=0)
        magnitude = float(torch.sqrt((c_real ** 2 + c_imag ** 2).sum()))
        if magnitude > max_magnitude:
            max_magnitude = magnitude
            dominant_k = k

    return max_magnitude, dominant_k


@torch.no_grad()
def compute_weight_norm_ratio(model, p):
    """Compute ||W_fourier|| / ||W_total||.

    Project the embedding weights into a Fourier basis and measure what
    fraction of the total weight norm is aligned with Fourier directions.
    """
    W_E = model.embed.W_E.data.float().cpu()  # (d_vocab, d_model)
    # Use only the first p rows (token embeddings, excluding the = token)
    W_tokens = W_E[:p, :]  # (p, d_model)

    total_norm = float(W_tokens.norm())
    if total_norm < 1e-10:
        return 0.0

    # Build Fourier basis: cos(2pi k i / p) and sin(2pi k i / p) for k=1..p//2
    fourier_norms_sq = 0.0
    indices = torch.arange(p, dtype=torch.float32)
    max_k = p // 2

    for k in range(1, max_k + 1):
        phases = 2.0 * math.pi * k * indices / p  # (p,)
        cos_basis = torch.cos(phases)  # (p,)
        sin_basis = torch.sin(phases)
        # Normalize basis vectors
        cos_basis = cos_basis / cos_basis.norm()
        sin_basis = sin_basis / sin_basis.norm()
        # Project each column of W_tokens onto cos and sin basis
        for col in range(W_tokens.shape[1]):
            w_col = W_tokens[:, col]
            proj_cos = float(torch.dot(w_col, cos_basis)) ** 2
            proj_sin = float(torch.dot(w_col, sin_basis)) ** 2
            fourier_norms_sq += proj_cos + proj_sin

    fourier_norm = math.sqrt(fourier_norms_sq)
    return fourier_norm / total_norm


def find_grokking_epoch(test_losses, threshold=0.1):
    """Find the first epoch where test loss drops below threshold.

    Returns the epoch index or None if it never crosses.
    """
    for i, loss in enumerate(test_losses):
        if loss < threshold:
            return i
    return None


def find_signal_fire_epoch(trajectory, signal_name, threshold):
    """Find the first epoch in the trajectory where the signal crosses threshold.

    For loss_ratio, the signal fires when the value drops BELOW threshold
    (memorization ending). For all others, fires when value rises ABOVE threshold.
    """
    for step in trajectory:
        val = step.get(signal_name)
        if val is None:
            continue
        if signal_name == "loss_ratio":
            if val < threshold:
                return step["epoch"]
        else:
            if val > threshold:
                return step["epoch"]
    return None


# ── Per-operation pipeline ──

def run_operation(operation, p, n_epochs, ckpt_interval, das_steps, device,
                  output_path, wandb_run, seed=42):
    """Train with dense checkpoints, measure all signals at each, analyze ordering.

    Appends one JSONL row and returns the result dict.
    """
    logger.info("[%s] === %s mod %d ===", utc_ts(), operation, p)

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
        p, device, n_epochs=n_epochs,
        checkpoint_every=ckpt_interval,
        dataset=dataset, labels=labels,
        train_idx=train_idx, test_idx=test_idx,
        seed=seed,
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
        logger.warning("  No checkpoints saved -- skipping")
        return None

    model.eval()
    d = cfg.d_model
    layer = 0
    k = 2
    site = site_resid(layer, d)

    # ── Measure all signals at each checkpoint ──
    trajectory = []
    logger.info("[%s] Measuring signals at %d checkpoints...", utc_ts(), n_ckpts)

    for i in tqdm(range(n_ckpts), desc=f"signals ({operation})"):
        epoch = checkpoint_epochs[i]
        model.load_state_dict(checkpoints[i])
        model.eval()

        trl = train_losses[epoch] if epoch < len(train_losses) else None
        tl = test_losses[epoch] if epoch < len(test_losses) else None

        # 1. Spectral gap
        spectral_gap, _ = compute_spectral_gap(model, dataset, layer, device)

        # 2. Fourier magnitude
        fourier_mag, fourier_k = compute_fourier_magnitude(
            model, dataset, labels, layer, device, p)

        # 3. DAS IIA at k=2
        cached = cache_pairs(model, dataset, labels, train_idx, layer, device,
                             n_pairs=200)
        iia_k2 = 0.0
        if len(cached) >= 10:
            _, iia_k2 = train_das(model, site, cached, k, das_steps, device)

        # 4. Equivariance score
        equiv_result = equivariance_test(
            model, site,
            # Need a Q for equivariance -- reuse from DAS if available
            torch.eye(d, k, device=device) if len(cached) < 10
            else train_das(model, site, cached, k, das_steps, device)[0],
            dataset, labels, device, p,
            operation=operation, n_test=200,
        )
        equiv_frac = equiv_result["equivariant_fraction"]

        # 5. Weight norm ratio
        weight_ratio = compute_weight_norm_ratio(model, p)

        # 6. Loss ratio
        loss_ratio = float("inf")
        if tl is not None and trl is not None and trl > 1e-8:
            loss_ratio = tl / trl

        step_data = {
            "epoch": epoch,
            "train_loss": trl,
            "test_loss": tl,
            "spectral_gap": float(spectral_gap),
            "fourier_magnitude": float(fourier_mag),
            "iia_k2": float(iia_k2),
            "equivariance": float(equiv_frac),
            "weight_norm_ratio": float(weight_ratio),
            "loss_ratio": float(loss_ratio),
        }
        trajectory.append(step_data)

        if wandb_run is not None:
            import wandb
            log_dict = {f"{operation}/{key}": step_data[key]
                        for key in SIGNAL_NAMES if step_data[key] is not None}
            log_dict["epoch"] = epoch
            if trl is not None:
                log_dict[f"{operation}/train_loss"] = trl
            if tl is not None:
                log_dict[f"{operation}/test_loss"] = tl
            wandb_run.log(log_dict, step=epoch)

        logger.info("  Epoch %6d: gap=%.3f fourier=%.3f iia=%.3f equiv=%.3f "
                     "wnorm=%.3f lratio=%.2f",
                     epoch, spectral_gap, fourier_mag, iia_k2, equiv_frac,
                     weight_ratio, loss_ratio)

    # Free checkpoint memory
    del checkpoints

    # ── Analyze signal ordering ──
    grokking_epoch = find_grokking_epoch(test_losses)

    signal_fire_epochs = {}
    for sig_name in SIGNAL_NAMES:
        fire_ep = find_signal_fire_epoch(trajectory, sig_name,
                                          SIGNAL_THRESHOLDS[sig_name])
        signal_fire_epochs[sig_name] = fire_ep

    # Compute lead times (how many epochs before grokking does each signal fire)
    signal_lead_times = {}
    for sig_name in SIGNAL_NAMES:
        fire_ep = signal_fire_epochs[sig_name]
        if fire_ep is not None and grokking_epoch is not None:
            signal_lead_times[sig_name] = grokking_epoch - fire_ep
        else:
            signal_lead_times[sig_name] = None

    # Determine signal ordering (earliest to latest)
    fired_signals = [(name, ep) for name, ep in signal_fire_epochs.items()
                     if ep is not None]
    fired_signals.sort(key=lambda x: x[1])
    signal_ordering = [name for name, _ in fired_signals]

    earliest_signal = signal_ordering[0] if signal_ordering else None
    earliest_signal_epoch = (signal_fire_epochs[earliest_signal]
                            if earliest_signal else None)

    result = {
        "operation": operation,
        "grokked": grokked,
        "trajectory": trajectory,
        "grokking_epoch": grokking_epoch,
        "earliest_signal": earliest_signal,
        "earliest_signal_epoch": earliest_signal_epoch,
        "signal_lead_times": signal_lead_times,
        "signal_ordering": signal_ordering,
        "timestamp": utc_ts(),
    }

    with open(output_path, "a") as f:
        f.write(json.dumps(result, default=str) + "\n")
    logger.info("[%s] Result for %s written to %s", utc_ts(), operation,
                output_path)

    # W&B summary
    if wandb_run is not None:
        import wandb
        summary = {
            f"{operation}/grokked": int(grokked),
            f"{operation}/grokking_epoch": grokking_epoch,
            f"{operation}/earliest_signal": earliest_signal,
            f"{operation}/earliest_signal_epoch": earliest_signal_epoch,
        }
        for sig_name, lead in signal_lead_times.items():
            if lead is not None:
                summary[f"{operation}/lead_time/{sig_name}"] = lead
        summary = {k: v for k, v in summary.items() if v is not None}
        wandb_run.log(summary)

    return result


# ── Main ──

def main():
    parser = argparse.ArgumentParser(
        description="Grokking onset predictors — which signal fires first?")
    parser.add_argument("--operations", required=True,
                        help="Comma-separated operations (or 'all')")
    parser.add_argument("--p", type=int, default=None,
                        help="Modulus (default: 113, or per-op override)")
    parser.add_argument("--n-epochs", type=int, default=None,
                        help="Override epoch count for all operations")
    parser.add_argument("--ckpt-interval", type=int, default=200,
                        help="Checkpoint every N epochs")
    parser.add_argument("--das-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("experiments/results"))
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    all_known_ops = list(EPOCH_DEFAULTS.keys())
    if args.operations == "all":
        operations = all_known_ops
    else:
        operations = [op.strip() for op in args.operations.split(",")]

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "grokking_onset_predictors.jsonl"

    logger.info("[%s] Grokking Onset Predictors -- %d operations",
                utc_ts(), len(operations))
    logger.info("  Operations: %s", operations)
    logger.info("  ckpt_interval: %d, DAS steps: %d, seed: %d",
                args.ckpt_interval, args.das_steps, args.seed)

    # W&B init
    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb
            run_name = (f"grokking_onset_{len(operations)}ops_"
                        f"p{args.p or 113}")
            wandb_run = wandb.init(
                project="SAELensCircuitPort - Experimental",
                entity="factorized-circuits",
                name=run_name,
                config={
                    "experiment": "grokking_onset_predictors",
                    "operations": operations,
                    "p": args.p,
                    "n_epochs": args.n_epochs,
                    "ckpt_interval": args.ckpt_interval,
                    "das_steps": args.das_steps,
                    "seed": args.seed,
                    "signal_thresholds": SIGNAL_THRESHOLDS,
                },
            )
        except Exception as e:
            logger.warning("W&B init failed: %s -- continuing without W&B", e)

    results = []
    for op in operations:
        p = args.p if args.p is not None else P_OVERRIDES.get(op, 113)
        n_epochs = args.n_epochs or EPOCH_DEFAULTS.get(op, 25000)

        result = run_operation(
            operation=op,
            p=p,
            n_epochs=n_epochs,
            ckpt_interval=args.ckpt_interval,
            das_steps=args.das_steps,
            device=args.device,
            output_path=out_file,
            wandb_run=wandb_run,
            seed=args.seed,
        )
        if result is not None:
            results.append(result)

    # ── Summary table ──
    print()
    print("=" * 100)
    print("Grokking Onset Predictors -- Summary")
    print("=" * 100)

    if not results:
        print("  No results.")
        if wandb_run is not None:
            import wandb
            wandb_run.finish(exit_code=1)
        return

    print(f"\n  {'Operation':<22} {'Grokked':>7} {'GrokEp':>8} "
          f"{'Earliest':>16} {'EarlEp':>8} {'#Fired':>7}")
    print(f"  {'-'*22} {'-'*7} {'-'*8} {'-'*16} {'-'*8} {'-'*7}")
    for r in results:
        grok_str = "YES" if r["grokked"] else "no"
        grok_ep = (f"{r['grokking_epoch']:8d}" if r["grokking_epoch"] is not None
                   else "     N/A")
        earliest = r["earliest_signal"] or "N/A"
        earl_ep = (f"{r['earliest_signal_epoch']:8d}"
                   if r["earliest_signal_epoch"] is not None else "     N/A")
        n_fired = len(r["signal_ordering"])
        print(f"  {r['operation']:<22} {grok_str:>7} {grok_ep} "
              f"{earliest:>16} {earl_ep} {n_fired:>7}")

    # Lead time analysis
    grokked_results = [r for r in results if r["grokked"]]
    if grokked_results:
        print(f"\n  Signal lead times (epochs before grokking, grokked ops only):")
        print(f"  {'Signal':<20} {'Mean':>10} {'Std':>10} {'Min':>10} "
              f"{'Max':>10} {'Count':>7}")
        print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*7}")

        for sig_name in SIGNAL_NAMES:
            leads = [r["signal_lead_times"][sig_name] for r in grokked_results
                     if r["signal_lead_times"][sig_name] is not None]
            if leads:
                mean_lead = np.mean(leads)
                std_lead = np.std(leads)
                print(f"  {sig_name:<20} {mean_lead:10.1f} {std_lead:10.1f} "
                      f"{min(leads):10d} {max(leads):10d} {len(leads):>7}")
            else:
                print(f"  {sig_name:<20} {'N/A':>10} {'N/A':>10} "
                      f"{'N/A':>10} {'N/A':>10} {'0':>7}")

    # Signal ordering frequency
    if grokked_results:
        print(f"\n  How often each signal fires FIRST (grokked ops):")
        first_counts = {}
        for r in grokked_results:
            if r["earliest_signal"]:
                first_counts[r["earliest_signal"]] = (
                    first_counts.get(r["earliest_signal"], 0) + 1)
        for sig_name, count in sorted(first_counts.items(),
                                       key=lambda x: -x[1]):
            frac = count / len(grokked_results)
            print(f"    {sig_name:<20} {count:3d} / {len(grokked_results)} "
                  f"({frac:.1%})")

    # Full signal ordering for each operation
    print(f"\n  Signal ordering per operation:")
    for r in results:
        ordering_str = " > ".join(r["signal_ordering"]) if r["signal_ordering"] else "none fired"
        grok_str = "GROK" if r["grokked"] else "NO  "
        print(f"    [{grok_str}] {r['operation']:<22} {ordering_str}")

    print("=" * 100)

    if wandb_run is not None:
        import wandb
        # Log summary statistics
        if grokked_results:
            first_counts = {}
            for r in grokked_results:
                if r["earliest_signal"]:
                    first_counts[r["earliest_signal"]] = (
                        first_counts.get(r["earliest_signal"], 0) + 1)
            for sig_name, count in first_counts.items():
                wandb_run.log({
                    f"summary/first_signal_count/{sig_name}": count,
                    f"summary/first_signal_frac/{sig_name}": (
                        count / len(grokked_results)),
                })

        artifact = wandb.Artifact("grokking-onset-predictors-results",
                                   type="geometry_wild_results")
        artifact.add_file(str(out_file))
        wandb_run.log_artifact(artifact)
        wandb_run.finish()

    logger.info("[%s] Done. Results in %s", utc_ts(), out_file)


if __name__ == "__main__":
    main()
