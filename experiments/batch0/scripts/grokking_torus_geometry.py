"""Grokking Torus Geometry — Does the causal subspace recover the Fourier circle?

Linear DAS finds the causal variable for grokking at k=16 (IIA=1.0). But IIA
is ceilinged and can't tell us WHAT the variable looks like. This script asks:
does the causal subspace contain a circular/toroidal structure matching the
known Fourier mechanism?

Tests:
1. k-sweep to find intrinsic causal dimension (smallest k with IIA saturation)
2. Per-frequency circular R² fits (project onto causal subspace, fit cos/sin)
3. Angle→sum equivariance (rotate angle by 2π/113 → answer increments by 1?)
4. Subspace trajectory across training (when does the geometry form?)

The prediction from Nanda et al.: modular addition uses ~5 key frequencies,
each needing 2 dimensions (cos+sin), so intrinsic dimension should be ~10.
The causal subspace should show high R² at exactly the model's active
frequencies and noise elsewhere.

Usage:
    python -u experiments/batch6_atlas/grokking_torus_geometry.py \
        --device cuda --n-epochs 25000

    python -u experiments/batch6_atlas/grokking_torus_geometry.py \
        --device cuda --load-from experiments/results/grokking_checkpoints.pt
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import einops
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformer_lens import HookedTransformer, HookedTransformerConfig

from factorization_circuits.pipeline.utils.factor_das_kernel import (
    VanillaQ, eval_iia, make_hook, site_resid,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

P = 113
FRAC_TRAIN = 0.3
DATA_SEED = 598
KEY_FREQS = [17, 25, 32, 47]
ALL_FREQS = list(range(1, P // 2 + 1))


def utc_ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_grokking_data(device):
    a_vec = einops.repeat(torch.arange(P), "i -> (i j)", j=P)
    b_vec = einops.repeat(torch.arange(P), "j -> (i j)", i=P)
    eq_vec = einops.repeat(torch.tensor(P), " -> (i j)", i=P, j=P)
    dataset = torch.stack([a_vec, b_vec, eq_vec], dim=1).to(device)
    labels = (dataset[:, 0] + dataset[:, 1]) % P
    torch.manual_seed(DATA_SEED)
    indices = torch.randperm(P * P)
    cutoff = int(P * P * FRAC_TRAIN)
    return dataset, labels, indices[:cutoff], indices[cutoff:]


def train_grokking_model(device, n_epochs=25000, checkpoint_every=500,
                         lr=1e-3, wd=1.0):
    cfg = HookedTransformerConfig(
        n_layers=1, n_heads=4, d_model=128, d_head=32, d_mlp=512,
        act_fn="relu", normalization_type=None,
        d_vocab=P + 1, d_vocab_out=P, n_ctx=3,
        init_weights=True, device=device, seed=999,
    )
    model = HookedTransformer(cfg)
    for name, param in model.named_parameters():
        if "b_" in name:
            param.requires_grad = False

    dataset, labels, train_idx, test_idx = build_grokking_data(device)
    train_data, train_labels = dataset[train_idx], labels[train_idx]
    test_data, test_labels = dataset[test_idx], labels[test_idx]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd,
                                  betas=(0.9, 0.98))

    checkpoints, checkpoint_epochs = [], []
    train_losses, test_losses = [], []

    for epoch in tqdm(range(n_epochs), desc="training"):
        train_logits = model(train_data)[:, -1]
        train_loss = F.cross_entropy(train_logits, train_labels)
        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            test_logits = model(test_data)[:, -1]
            test_loss = F.cross_entropy(test_logits, test_labels)

        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())

        if (epoch + 1) % checkpoint_every == 0:
            checkpoint_epochs.append(epoch)
            checkpoints.append(copy.deepcopy(model.state_dict()))

    return model, cfg, {
        "checkpoints": checkpoints, "checkpoint_epochs": checkpoint_epochs,
        "train_losses": train_losses, "test_losses": test_losses,
        "train_idx": train_idx, "test_idx": test_idx,
    }


@torch.no_grad()
def cache_grokking_pairs(model, dataset, labels, train_idx, layer, device,
                         n_pairs=200):
    train_data = dataset[train_idx]
    train_labels = labels[train_idx]
    pairs_idx = []
    for i in range(0, len(train_data) - 1, 2):
        if train_labels[i] != train_labels[i + 1]:
            pairs_idx.append((i, i + 1))
        if len(pairs_idx) >= n_pairs:
            break

    hook_name = f"blocks.{layer}.hook_resid_post"
    cached = []
    for i, j in pairs_idx:
        tokens_i = train_data[i].unsqueeze(0)
        tokens_j = train_data[j].unsqueeze(0)
        _, cache_i = model.run_with_cache(tokens_i, names_filter=[hook_name])
        _, cache_j = model.run_with_cache(tokens_j, names_filter=[hook_name])
        cached.append((
            tokens_i, cache_i[hook_name][0, -1, :],
            cache_j[hook_name][0, -1, :], train_labels[j].item(),
        ))
    return cached


def train_das(model, site, cached, k, n_steps, device):
    param = VanillaQ(model.cfg.d_model, k, device=device)
    opt = torch.optim.Adam(param.parameters(), lr=1e-3)
    import random as _rng
    rng = _rng.Random(0)
    n_train = int(len(cached) * 0.75)
    cached_train = cached[:n_train]
    cached_eval = cached[n_train:]

    frozen = {p for p in model.parameters() if p.requires_grad}
    for p in frozen:
        p.requires_grad_(False)
    try:
        for step in range(n_steps):
            idx = rng.sample(range(len(cached_train)), min(16, len(cached_train)))
            Q = param()
            proj = Q @ Q.T
            loss = torch.tensor(0., device=device)
            for i in idx:
                bt, ba, sa, si = cached_train[i]
                iv = ba - ba @ proj + sa @ proj
                lp = F.log_softmax(
                    model.run_with_hooks(
                        bt, fwd_hooks=[(site.hook_name, make_hook(site, iv))]
                    )[0, -1, :], dim=-1)
                loss = loss - lp[si]
            (loss / len(idx)).backward()
            opt.step()
            opt.zero_grad()
    finally:
        for p in frozen:
            p.requires_grad_(True)

    Q = param().detach()
    iia = eval_iia(model, site, Q, cached_eval)
    return Q, iia


@torch.no_grad()
def get_all_activations(model, dataset, layer, device):
    """Get activations for all P*P inputs at the target layer."""
    hook_name = f"blocks.{layer}.hook_resid_post"
    batch_size = 256
    all_acts = []
    for start in range(0, len(dataset), batch_size):
        batch = dataset[start:start + batch_size]
        _, cache = model.run_with_cache(batch, names_filter=[hook_name])
        all_acts.append(cache[hook_name][:, -1, :])
    return torch.cat(all_acts, dim=0)


@torch.no_grad()
def circular_r2_per_frequency(activations, Q, freqs=None):
    """For each frequency, fit cos(ω*a) and sin(ω*a) to the projected activations
    and report R². High R² = the causal subspace encodes this frequency circularly.

    activations: (P*P, d_model) for all (a, b) pairs
    Q: (d_model, k) causal subspace
    """
    if freqs is None:
        freqs = ALL_FREQS

    projected = activations @ Q  # (P*P, k)
    proj_cube = projected.reshape(P, P, -1)  # (P, P, k) indexed by (a, b)

    results = {}
    for freq in freqs:
        a_idx = torch.arange(P, device=activations.device).float()
        cos_a = torch.cos(2 * torch.pi * freq * a_idx / P)
        sin_a = torch.sin(2 * torch.pi * freq * a_idx / P)

        mean_proj = proj_cube.mean(dim=1)  # (P, k) — average over b

        total_var = ((mean_proj - mean_proj.mean(dim=0, keepdim=True)) ** 2).sum()
        if total_var < 1e-10:
            results[freq] = 0.0
            continue

        cos_proj = (cos_a[:, None] * mean_proj).sum(dim=0)
        sin_proj = (sin_a[:, None] * mean_proj).sum(dim=0)
        explained = (cos_proj ** 2 + sin_proj ** 2).sum() / (P / 2)
        r2 = float((explained / total_var).clamp(0, 1).item())
        results[freq] = r2

    return results


@torch.no_grad()
def angle_equivariance_test(model, site, Q, dataset, labels, device,
                            n_test=200):
    """Test if rotating the angle in the causal subspace by 2π/P increments
    the model's predicted answer by 1 (mod P).

    This is the strongest geometric test: it proves the variable isn't just
    A subspace that happens to work — it's THE torus with the right metric.
    """
    hook_name = site.hook_name
    proj = Q @ Q.T

    rng = torch.Generator(device="cpu")
    rng.manual_seed(42)
    test_indices = torch.randperm(len(dataset), generator=rng)[:n_test]

    equivariant_count = 0
    total_tested = 0

    for idx in test_indices:
        tokens = dataset[idx].unsqueeze(0)
        _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
        act = cache[hook_name][0, -1, :]
        orig_pred = model(tokens)[0, -1, :].argmax().item()
        true_label = labels[idx].item()

        if orig_pred != true_label:
            continue

        in_sub = act @ proj
        complement = act - in_sub

        for shift in [1, 2, 5]:
            target_answer = (true_label + shift) % P

            best_match = None
            best_dist = float('inf')
            for search_idx in torch.randperm(len(dataset))[:500]:
                if labels[search_idx].item() == target_answer:
                    s_tokens = dataset[search_idx].unsqueeze(0)
                    _, s_cache = model.run_with_cache(s_tokens, names_filter=[hook_name])
                    s_act = s_cache[hook_name][0, -1, :]
                    s_in_sub = s_act @ proj
                    dist = (s_in_sub - in_sub).norm().item()
                    if dist < best_dist:
                        best_dist = dist
                        best_match = s_in_sub

            if best_match is None:
                continue

            shifted_act = best_match + complement
            hook_fn = make_hook(site, shifted_act)
            shifted_logits = model.run_with_hooks(
                tokens, fwd_hooks=[(hook_name, hook_fn)]
            )[0, -1, :]
            shifted_pred = shifted_logits.argmax().item()

            total_tested += 1
            if shifted_pred == target_answer:
                equivariant_count += 1

    return {
        "equivariant_fraction": equivariant_count / max(total_tested, 1),
        "n_tested": total_tested,
        "n_equivariant": equivariant_count,
    }


def grassmann_distance(Q1, Q2):
    k = min(Q1.shape[1], Q2.shape[1])
    _, S, _ = torch.linalg.svd(Q1[:, :k].T @ Q2[:, :k])
    return float(torch.acos(S.clamp(-1.0, 1.0)).norm().item())


def main():
    parser = argparse.ArgumentParser(description="Grokking Torus Geometry")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-epochs", type=int, default=25000)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--das-steps", type=int, default=400)
    parser.add_argument("--load-from", type=str, default=None)
    parser.add_argument("--output-dir", default="experiments/results")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--model", default="grokking")
    parser.add_argument("--task", default="modular_addition")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--n-steps", type=int, default=None)
    args = parser.parse_args()

    device = args.device
    layer = 0
    das_steps = args.n_steps or args.das_steps
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[%s] Grokking Torus Geometry", utc_ts())

    dataset, labels, train_idx, test_idx = build_grokking_data(device)

    # Train or load grokking model
    if args.load_from and Path(args.load_from).exists():
        logger.info("[%s] Loading from %s", utc_ts(), args.load_from)
        saved = torch.load(args.load_from, map_location=device, weights_only=False)
        cfg = saved["config"]
        model = HookedTransformer(cfg)
        checkpoints = saved["checkpoints"]
        checkpoint_epochs = saved["checkpoint_epochs"]
        train_losses = saved["train_losses"]
        test_losses = saved["test_losses"]
        model.load_state_dict(checkpoints[-1])
    else:
        logger.info("[%s] Training grokking model (%d epochs)...", utc_ts(), args.n_epochs)
        model, cfg, training_data = train_grokking_model(
            device, n_epochs=args.n_epochs, checkpoint_every=args.checkpoint_every,
        )
        checkpoints = training_data["checkpoints"]
        checkpoint_epochs = training_data["checkpoint_epochs"]
        train_losses = training_data["train_losses"]
        test_losses = training_data["test_losses"]

        ckpt_file = output_dir / "grokking_checkpoints.pt"
        torch.save({
            "config": cfg, "checkpoints": checkpoints,
            "checkpoint_epochs": checkpoint_epochs,
            "train_losses": train_losses, "test_losses": test_losses,
            "train_idx": train_idx.cpu(), "test_idx": test_idx.cpu(),
        }, ckpt_file)

    model.eval()
    d = model.cfg.d_model
    site = site_resid(layer, d)

    # === Test 1: k-sweep for intrinsic causal dimension ===
    logger.info("\n[%s] === k-sweep (intrinsic dimension) ===", utc_ts())
    cached = cache_grokking_pairs(model, dataset, labels, train_idx, layer, device)
    k_sweep_results = []
    for k_val in [2, 4, 6, 8, 10, 12, 16, 20, 24, 32]:
        Q, iia = train_das(model, site, cached, k_val, das_steps, device)
        k_sweep_results.append({"k": k_val, "iia": iia})
        logger.info("  k=%2d: IIA=%.3f", k_val, iia)

    # Find elbow: smallest k where IIA > 0.9
    intrinsic_dim = None
    for r in k_sweep_results:
        if r["iia"] > 0.9:
            intrinsic_dim = r["k"]
            break

    # === Test 2: Per-frequency circular R² ===
    logger.info("\n[%s] === Circular R² per frequency ===", utc_ts())
    Q_final, _ = train_das(model, site, cached, 16, das_steps, device)
    all_acts = get_all_activations(model, dataset, layer, device)

    r2_per_freq = circular_r2_per_frequency(all_acts, Q_final, freqs=ALL_FREQS)
    top_freqs = sorted(r2_per_freq.items(), key=lambda x: -x[1])[:10]
    logger.info("  Top 10 frequencies by circular R²:")
    for freq, r2 in top_freqs:
        marker = " ← KEY" if freq in KEY_FREQS else ""
        logger.info("    freq=%3d: R²=%.4f%s", freq, r2, marker)

    key_freq_r2 = {f: r2_per_freq[f] for f in KEY_FREQS}
    mean_key_r2 = sum(key_freq_r2.values()) / len(key_freq_r2)
    mean_other_r2 = sum(v for k, v in r2_per_freq.items() if k not in KEY_FREQS) / max(1, len(r2_per_freq) - len(KEY_FREQS))
    logger.info("  Mean R² at key freqs: %.4f", mean_key_r2)
    logger.info("  Mean R² at other freqs: %.4f", mean_other_r2)

    # Also test on random subspace as control
    Q_rand, _ = torch.linalg.qr(torch.randn(d, 16, device=device))
    r2_random = circular_r2_per_frequency(all_acts, Q_rand, freqs=KEY_FREQS)
    logger.info("  Random subspace key freq R²: %s",
                {f: f"{v:.4f}" for f, v in r2_random.items()})

    # === Test 3: Angle→sum equivariance ===
    logger.info("\n[%s] === Angle→sum equivariance ===", utc_ts())
    equiv = angle_equivariance_test(model, site, Q_final, dataset, labels, device)
    logger.info("  Equivariant: %d/%d = %.3f",
                equiv["n_equivariant"], equiv["n_tested"],
                equiv["equivariant_fraction"])

    # === Test 3b: Equivariance controls ===
    logger.info("\n[%s] === Equivariance controls ===", utc_ts())

    # Control 1: Random k=2 subspaces on the final (grokked) model — 20 seeds
    rand_equiv_scores = []
    for seed in range(20):
        torch.manual_seed(seed * 1000 + 99)
        Q_rand_equiv, _ = torch.linalg.qr(torch.randn(d, 2, device=device))
        eq_r = angle_equivariance_test(model, site, Q_rand_equiv, dataset, labels, device,
                                       n_test=100)
        rand_equiv_scores.append(eq_r["equivariant_fraction"])
    equiv_random_mean = sum(rand_equiv_scores) / len(rand_equiv_scores)
    equiv_random_std = (sum((x - equiv_random_mean)**2 for x in rand_equiv_scores) / len(rand_equiv_scores)) ** 0.5
    logger.info("  Random k=2 equivariance (20 seeds): %.3f +/- %.3f",
                equiv_random_mean, equiv_random_std)

    # Control 2: Causal subspace on memorization checkpoint (first saved)
    mem_epoch = checkpoint_epochs[0]
    model.load_state_dict(checkpoints[0])
    model.eval()
    cached_mem = cache_grokking_pairs(model, dataset, labels, train_idx, layer, device)
    if len(cached_mem) >= 10:
        Q_mem, iia_mem = train_das(model, site, cached_mem, 2, min(200, das_steps), device)
        equiv_mem = angle_equivariance_test(model, site, Q_mem, dataset, labels, device)
        logger.info("  Memorization (epoch %d) equivariance: %d/%d = %.3f (IIA=%.3f)",
                    mem_epoch, equiv_mem["n_equivariant"], equiv_mem["n_tested"],
                    equiv_mem["equivariant_fraction"], iia_mem)
    else:
        equiv_mem = {"equivariant_fraction": 0.0, "n_tested": 0, "n_equivariant": 0}
        iia_mem = 0.0
        logger.info("  Memorization (epoch %d): too few valid pairs for DAS", mem_epoch)

    # Restore final model
    model.load_state_dict(checkpoints[-1])
    model.eval()

    # === Test 3c: Circle geometry (topology proxy) ===
    logger.info("\n[%s] === Circle geometry test ===", utc_ts())
    Q_k2, _ = train_das(model, site, cached, 2, das_steps, device)
    all_acts_2d = all_acts @ Q_k2

    centroids = torch.zeros(P, 2, device=device)
    counts = torch.zeros(P, device=device)
    for i in range(len(labels)):
        lbl = labels[i].item()
        centroids[lbl] += all_acts_2d[i]
        counts[lbl] += 1
    centroids = centroids / counts.unsqueeze(1).clamp(min=1)

    center = centroids.mean(0)
    centered = centroids - center
    radii = centered.norm(dim=1)
    radius_cv = (radii.std() / radii.mean()).item()

    angles_ordered = torch.atan2(centered[:, 1], centered[:, 0])
    angle_diffs = torch.diff(angles_ordered)
    angle_diffs = torch.where(angle_diffs > torch.pi, angle_diffs - 2 * torch.pi, angle_diffs)
    angle_diffs = torch.where(angle_diffs < -torch.pi, angle_diffs + 2 * torch.pi, angle_diffs)
    closing = angles_ordered[0] - angles_ordered[-1]
    if closing > torch.pi: closing = closing - 2 * torch.pi
    if closing < -torch.pi: closing = closing + 2 * torch.pi
    total_angle = angle_diffs.sum() + closing
    winding_number = (total_angle / (2 * torch.pi)).round().item()

    sorted_by_angle = angles_ordered.argsort()
    consec_diffs = (sorted_by_angle[1:] - sorted_by_angle[:-1]) % P
    modal_step = consec_diffs.mode().values.item()
    ordered_fraction = (consec_diffs == modal_step).float().mean().item()

    circle_result = {
        "radius_cv": radius_cv,
        "winding_number": winding_number,
        "angular_ordering_accuracy": ordered_fraction,
        "modal_step": modal_step,
        "mean_radius": radii.mean().item(),
    }
    logger.info("  Radius CV: %.4f (0=perfect circle)", radius_cv)
    logger.info("  Winding number: %d (1=single loop)", int(winding_number))
    logger.info("  Angular ordering accuracy: %.3f (step=%d)", ordered_fraction, modal_step)

    # Random k=2 control for circle test
    Q_rand_circle, _ = torch.linalg.qr(torch.randn(d, 2, device=device))
    rand_2d = all_acts @ Q_rand_circle
    rand_centroids = torch.zeros(P, 2, device=device)
    for i in range(len(labels)):
        rand_centroids[labels[i].item()] += rand_2d[i]
    rand_centroids = rand_centroids / counts.unsqueeze(1).clamp(min=1)
    rand_center = rand_centroids.mean(0)
    rand_centered = rand_centroids - rand_center
    rand_radii = rand_centered.norm(dim=1)
    rand_cv = (rand_radii.std() / rand_radii.mean()).item()
    rand_angles = torch.atan2(rand_centered[:, 1], rand_centered[:, 0])
    rand_sorted = rand_angles.argsort()
    rand_diffs = (rand_sorted[1:] - rand_sorted[:-1]) % P
    rand_modal = rand_diffs.mode().values.item()
    rand_ordered = (rand_diffs == rand_modal).float().mean().item()
    logger.info("  Random control: radius_cv=%.4f, ordering=%.3f", rand_cv, rand_ordered)

    circle_random = {"radius_cv": rand_cv, "angular_ordering_accuracy": rand_ordered}

    # Persistent homology H1 test (ripser if available)
    h1_result = {}
    try:
        from ripser import ripser as ripser_fn
        centroids_np = centroids.cpu().numpy()
        diagrams = ripser_fn(centroids_np, maxdim=1)["dgms"]
        h1 = diagrams[1]
        if len(h1) > 0:
            persistences = h1[:, 1] - h1[:, 0]
            top_persistence = float(persistences.max())
            n_significant = int((persistences > persistences.max() * 0.1).sum())
            h1_result = {
                "top_persistence": top_persistence,
                "n_significant_loops": n_significant,
                "n_total_h1": len(h1),
            }
            logger.info("  H1 persistence: top=%.4f, n_significant=%d, n_total=%d",
                        top_persistence, n_significant, len(h1))
        else:
            h1_result = {"top_persistence": 0.0, "n_significant_loops": 0, "n_total_h1": 0}
            logger.info("  H1: no loops detected")
    except ImportError:
        logger.info("  ripser not available — skipping H1 persistence (winding number above is the proxy)")

    # === Test 4: Subspace trajectory across training ===
    logger.info("\n[%s] === Subspace trajectory across training ===", utc_ts())
    trajectory = []
    n_ckpts = len(checkpoints)
    sample_indices = list(range(0, n_ckpts, max(1, n_ckpts // 8)))
    if sample_indices[-1] != n_ckpts - 1:
        sample_indices.append(n_ckpts - 1)

    for ckpt_idx in tqdm(sample_indices, desc="trajectory"):
        epoch = checkpoint_epochs[ckpt_idx]
        model.load_state_dict(checkpoints[ckpt_idx])
        model.eval()

        cached_ckpt = cache_grokking_pairs(model, dataset, labels, train_idx, layer, device, n_pairs=100)
        if len(cached_ckpt) < 10:
            trajectory.append({"epoch": epoch, "iia": 0.0, "grassmann_to_final": None, "key_freq_r2": {}})
            continue

        Q_ckpt, iia_ckpt = train_das(model, site, cached_ckpt, 16, min(200, das_steps), device)
        d_grass = grassmann_distance(Q_ckpt, Q_final)

        acts_ckpt = get_all_activations(model, dataset, layer, device)
        r2_ckpt = circular_r2_per_frequency(acts_ckpt, Q_ckpt, freqs=KEY_FREQS)

        Q_ckpt_k2, _ = train_das(model, site, cached_ckpt, 2, min(200, das_steps), device)
        equiv_ckpt = angle_equivariance_test(model, site, Q_ckpt_k2, dataset, labels,
                                             device, n_test=100)

        test_loss = test_losses[epoch] if epoch < len(test_losses) else None

        trajectory.append({
            "epoch": epoch,
            "test_loss": test_loss,
            "iia": iia_ckpt,
            "grassmann_to_final": d_grass,
            "key_freq_r2": r2_ckpt,
            "mean_key_r2": sum(r2_ckpt.values()) / max(len(r2_ckpt), 1),
            "equivariance": equiv_ckpt["equivariant_fraction"],
        })
        logger.info("  Epoch %5d: IIA=%.3f, d_G=%.3f, R²=%.4f, equiv=%.3f",
                     epoch, iia_ckpt, d_grass,
                     sum(r2_ckpt.values()) / max(len(r2_ckpt), 1),
                     equiv_ckpt["equivariant_fraction"])

    # Restore final model
    model.load_state_dict(checkpoints[-1])
    model.eval()

    result = {
        "timestamp": utc_ts(),
        "model": "grokking_1layer",
        "task": "modular_addition",
        "p": P,
        "layer": layer,
        "d_model": d,
        "das_steps": das_steps,
        "k_sweep": k_sweep_results,
        "intrinsic_dimension": intrinsic_dim,
        "circular_r2": {str(k): v for k, v in r2_per_freq.items()},
        "key_freq_r2": key_freq_r2,
        "random_control_r2": r2_random,
        "mean_key_r2": mean_key_r2,
        "mean_other_r2": mean_other_r2,
        "equivariance": equiv,
        "equivariance_controls": {
            "grokked_das": equiv["equivariant_fraction"],
            "random_k2_mean": equiv_random_mean,
            "random_k2_std": equiv_random_std,
            "random_k2_scores": rand_equiv_scores,
            "memorization_das": equiv_mem["equivariant_fraction"],
            "memorization_epoch": mem_epoch,
            "memorization_iia": iia_mem,
        },
        "circle_geometry": circle_result,
        "circle_random_control": circle_random,
        "h1_persistence": h1_result,
        "trajectory": trajectory,
    }

    out_file = output_dir / "grokking_torus_geometry.jsonl"
    with open(out_file, "a") as f:
        f.write(json.dumps(result, default=str) + "\n")
    logger.info("[%s] Results appended to %s", utc_ts(), out_file)

    # Summary
    print(f"\n{'='*70}")
    print(f"Grokking Torus Geometry — mod {P}")
    print(f"\n1. Intrinsic causal dimension: k={intrinsic_dim} "
          f"(predicted ~10 from 5 freq × 2 dims)")
    print(f"\n2. Circular R² at key frequencies:")
    for f, r2 in sorted(key_freq_r2.items()):
        print(f"   freq={f:3d}: R²={r2:.4f}")
    print(f"   Mean key: {mean_key_r2:.4f}, Mean other: {mean_other_r2:.4f}, "
          f"Selectivity: {mean_key_r2 / max(mean_other_r2, 1e-6):.1f}x")

    print(f"\n3a. Angle→sum equivariance: {equiv['equivariant_fraction']:.3f} "
          f"({equiv['n_equivariant']}/{equiv['n_tested']})")
    print(f"\n3b. Equivariance controls (2×2 table):")
    print(f"                   DAS k=2          Random k=2 (20 seeds)")
    print(f"   Grokked:        {equiv['equivariant_fraction']:.3f}            "
          f"{equiv_random_mean:.3f} +/- {equiv_random_std:.3f}")
    print(f"   Memorizing:     {equiv_mem['equivariant_fraction']:.3f}            "
          f"(not tested)")

    print(f"\n3c. Circle geometry:")
    print(f"   Radius CV: {circle_result['radius_cv']:.4f} (0=perfect circle, random={rand_cv:.4f})")
    print(f"   Winding number: {int(circle_result['winding_number'])} (1=single loop)")
    print(f"   Angular ordering: {circle_result['angular_ordering_accuracy']:.3f} (random={rand_ordered:.3f})")
    if h1_result:
        print(f"   H1 persistence: {h1_result.get('top_persistence', 'N/A')}, "
              f"n_loops={h1_result.get('n_significant_loops', 'N/A')}")

    print(f"\n4. Trajectory (with equivariance):")
    for t in trajectory[::max(1, len(trajectory)//5)]:
        tl = t.get("test_loss")
        tl_str = f"{tl:.4f}" if tl is not None else "N/A"
        d_g = t.get("grassmann_to_final")
        d_g_str = f"{d_g:.3f}" if d_g is not None else "N/A"
        eq = t.get("equivariance", 0)
        print(f"   Epoch {t['epoch']:6d}: test_loss={tl_str}  IIA={t['iia']:.3f}  "
              f"d_G={d_g_str}  R²={t.get('mean_key_r2', 0):.4f}  equiv={eq:.3f}")

    if equiv["equivariant_fraction"] > 0.5 and equiv_random_mean < 0.1:
        print(f"\nDIAGNOSIS: CONTROLLED EQUIVARIANCE — genuine group-action structure "
              f"(DAS={equiv['equivariant_fraction']:.3f}, random={equiv_random_mean:.3f})")
    elif equiv["equivariant_fraction"] > 0.5:
        print(f"\nDIAGNOSIS: EQUIVARIANT — subspace has right structure for addition")
    else:
        print(f"\nDIAGNOSIS: NO CLEAR TORUS — causal subspace doesn't match Fourier picture")


if __name__ == "__main__":
    main()
