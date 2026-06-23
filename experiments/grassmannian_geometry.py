"""Grassmannian Geometry — analyze DAS subspaces as points on Gr(k, d).

Three analyses that go beyond "we ran DAS and got high IIA":

1. Principal angles between operations: geodesic distance on Gr(k, d) between
   the DAS subspaces found for different grokking operations. Proves they're
   geometrically distinct representations, not just "different IIA numbers."

2. Grassmannian trajectory during grokking: train DAS at each saved checkpoint,
   track the subspace as a path on Gr(k, d). The prediction: the subspace SNAPS
   to its final position during the memorization→generalization phase transition.

3. Basin width / sharpness: perturb Q along random tangent vectors of Gr(k, d),
   measure IIA degradation vs geodesic distance. Sharp basin = the subspace is
   a precise geometric object, not any random 2D plane.

Usage:
    python -u experiments/batch6_atlas/grassmannian_geometry.py \\
        --operations multiplication,composite_addition --device cuda

    python -u experiments/batch6_atlas/grassmannian_geometry.py \\
        --operations multiplication --device cuda --test trajectory
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

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformer_lens import HookedTransformer, HookedTransformerConfig

from factorization_circuits.pipeline.utils.factor_das_kernel import (
    VanillaQ, Site, eval_iia, make_hook,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


# ── Reused from grokking_nonlinear_hunt ──────────────────────────────

FRAC_TRAIN = 0.3

def build_data(operation, p, device):
    if operation == "multiplication":
        a_vals = torch.arange(1, p)
        b_vals = torch.arange(1, p)
        a_vec = a_vals.repeat_interleave(len(b_vals))
        b_vec = b_vals.repeat(len(a_vals))
        eq_vec = torch.full_like(a_vec, p)
        dataset = torch.stack([a_vec, b_vec, eq_vec], dim=1).to(device)
        labels = ((a_vec * b_vec) % p).to(device)
    elif operation == "composite_addition":
        a_vals = torch.arange(p)
        b_vals = torch.arange(p)
        a_vec = a_vals.repeat_interleave(len(b_vals))
        b_vec = b_vals.repeat(len(a_vals))
        eq_vec = torch.full_like(a_vec, p)
        dataset = torch.stack([a_vec, b_vec, eq_vec], dim=1).to(device)
        labels = ((a_vec + b_vec) % p).to(device)
    elif operation == "division":
        a_vals = torch.arange(1, p)
        b_vals = torch.arange(1, p)
        a_vec = a_vals.repeat_interleave(len(b_vals))
        b_vec = b_vals.repeat(len(a_vals))
        eq_vec = torch.full_like(a_vec, p)
        dataset = torch.stack([a_vec, b_vec, eq_vec], dim=1).to(device)
        b_inv = torch.tensor([pow(int(b.item()), p - 2, p) for b in b_vec])
        labels = ((a_vec * b_inv) % p).to(device)
    elif operation == "squaring":
        a_vals = torch.arange(1, p)
        pad = torch.zeros(len(a_vals), dtype=torch.long)
        eq_vec = torch.full((len(a_vals),), p, dtype=torch.long)
        dataset = torch.stack([a_vals, pad, eq_vec], dim=1).to(device)
        labels = ((a_vals * a_vals) % p).to(device)
    elif operation == "cubing":
        a_vals = torch.arange(1, p)
        pad = torch.zeros(len(a_vals), dtype=torch.long)
        eq_vec = torch.full((len(a_vals),), p, dtype=torch.long)
        dataset = torch.stack([a_vals, pad, eq_vec], dim=1).to(device)
        labels = ((a_vals * a_vals * a_vals) % p).to(device)
    elif operation == "polynomial":
        a_vals = torch.arange(p)
        b_vals = torch.arange(p)
        a_vec = a_vals.repeat_interleave(len(b_vals))
        b_vec = b_vals.repeat(len(a_vals))
        eq_vec = torch.full_like(a_vec, p)
        dataset = torch.stack([a_vec, b_vec, eq_vec], dim=1).to(device)
        labels = ((a_vec * a_vec + b_vec) % p).to(device)
    elif operation == "max_ab":
        a_vec = torch.arange(p).repeat_interleave(p)
        b_vec = torch.arange(p).repeat(p)
        eq_vec = torch.full((p * p,), p, dtype=torch.long)
        dataset = torch.stack([a_vec, b_vec, eq_vec], dim=1).to(device)
        labels = torch.max(a_vec, b_vec).to(device) % p
    elif operation == "abs_diff":
        a_vec = torch.arange(p).repeat_interleave(p)
        b_vec = torch.arange(p).repeat(p)
        eq_vec = torch.full((p * p,), p, dtype=torch.long)
        dataset = torch.stack([a_vec, b_vec, eq_vec], dim=1).to(device)
        labels = ((a_vec - b_vec).abs() % p).to(device)
    elif operation == "sum_of_squares":
        a_vec = torch.arange(p).repeat_interleave(p)
        b_vec = torch.arange(p).repeat(p)
        eq_vec = torch.full((p * p,), p, dtype=torch.long)
        dataset = torch.stack([a_vec, b_vec, eq_vec], dim=1).to(device)
        labels = ((a_vec * a_vec + b_vec * b_vec) % p).to(device)
    elif operation == "power":
        a_vals = torch.arange(1, p)
        b_vals = torch.arange(1, p)
        a_vec = a_vals.repeat_interleave(len(b_vals))
        b_vec = b_vals.repeat(len(a_vals))
        eq_vec = torch.full_like(a_vec, p)
        dataset = torch.stack([a_vec, b_vec, eq_vec], dim=1).to(device)
        labels = torch.tensor([pow(int(a.item()), int(b.item()), p)
                                for a, b in zip(a_vec, b_vec)]).to(device)
    elif operation == "shifted_mult":
        a_vals = torch.arange(p)
        b_vals = torch.arange(p)
        a_vec = a_vals.repeat_interleave(len(b_vals))
        b_vec = b_vals.repeat(len(a_vals))
        eq_vec = torch.full_like(a_vec, p)
        dataset = torch.stack([a_vec, b_vec, eq_vec], dim=1).to(device)
        labels = (((a_vec + 1) * (b_vec + 1) - 1) % p).to(device)
    elif operation == "min_ab":
        a_vec = torch.arange(p).repeat_interleave(p)
        b_vec = torch.arange(p).repeat(p)
        eq_vec = torch.full((p * p,), p, dtype=torch.long)
        dataset = torch.stack([a_vec, b_vec, eq_vec], dim=1).to(device)
        labels = torch.min(a_vec, b_vec).to(device) % p
    elif operation == "floor_div":
        a_vals = torch.arange(p)
        b_vals = torch.arange(1, p)
        a_vec = a_vals.repeat_interleave(len(b_vals))
        b_vec = b_vals.repeat(len(a_vals))
        eq_vec = torch.full_like(a_vec, p)
        dataset = torch.stack([a_vec, b_vec, eq_vec], dim=1).to(device)
        labels = (a_vec // b_vec % p).to(device)
    elif operation == "bitwise_xor":
        a_vec = torch.arange(p).repeat_interleave(p)
        b_vec = torch.arange(p).repeat(p)
        eq_vec = torch.full((p * p,), p, dtype=torch.long)
        dataset = torch.stack([a_vec, b_vec, eq_vec], dim=1).to(device)
        labels = ((a_vec ^ b_vec) % p).to(device)
    elif operation == "gcd":
        a_vals = torch.arange(1, p)
        b_vals = torch.arange(1, p)
        a_vec = a_vals.repeat_interleave(len(b_vals))
        b_vec = b_vals.repeat(len(a_vals))
        eq_vec = torch.full_like(a_vec, p)
        dataset = torch.stack([a_vec, b_vec, eq_vec], dim=1).to(device)
        labels = torch.tensor([int(torch.gcd(a, b).item()) % p
                               for a, b in zip(a_vec, b_vec)]).to(device)
    elif operation == "subtraction":
        a_vec = torch.arange(p).repeat_interleave(p)
        b_vec = torch.arange(p).repeat(p)
        eq_vec = torch.full((p * p,), p, dtype=torch.long)
        dataset = torch.stack([a_vec, b_vec, eq_vec], dim=1).to(device)
        labels = ((a_vec - b_vec) % p).to(device)
    elif operation == "affine":
        a_vec = torch.arange(p).repeat_interleave(p)
        b_vec = torch.arange(p).repeat(p)
        eq_vec = torch.full((p * p,), p, dtype=torch.long)
        dataset = torch.stack([a_vec, b_vec, eq_vec], dim=1).to(device)
        labels = ((2 * a_vec + 3 * b_vec + 5) % p).to(device)
    elif operation == "cubic_sum":
        a_vec = torch.arange(p).repeat_interleave(p)
        b_vec = torch.arange(p).repeat(p)
        eq_vec = torch.full((p * p,), p, dtype=torch.long)
        dataset = torch.stack([a_vec, b_vec, eq_vec], dim=1).to(device)
        labels = ((a_vec ** 3 + b_vec ** 3) % p).to(device)
    elif operation == "modular_distance":
        a_vec = torch.arange(p).repeat_interleave(p)
        b_vec = torch.arange(p).repeat(p)
        eq_vec = torch.full((p * p,), p, dtype=torch.long)
        dataset = torch.stack([a_vec, b_vec, eq_vec], dim=1).to(device)
        raw_diff = (a_vec - b_vec).abs()
        labels = torch.min(raw_diff, p - raw_diff).to(device)
    else:
        raise ValueError(f"Unknown operation: {operation}")
    return dataset, labels


def make_train_test_split(n, frac_train, seed=0):
    rng = torch.Generator()
    rng.manual_seed(seed)
    perm = torch.randperm(n, generator=rng)
    n_train = int(n * frac_train)
    return perm[:n_train], perm[n_train:]


def train_grokking_model(p, device, n_epochs=25000, checkpoint_every=500,
                         lr=1e-3, wd=1.0, dataset=None, labels=None,
                         train_idx=None, test_idx=None):
    cfg = HookedTransformerConfig(
        n_layers=1, n_heads=4, d_model=128, d_head=32, d_mlp=512,
        act_fn="relu", normalization_type=None,
        d_vocab=p + 1, d_vocab_out=p, n_ctx=3,
        init_weights=True, device=device, seed=999,
    )
    model = HookedTransformer(cfg)
    for name, param in model.named_parameters():
        if "b_" in name:
            param.requires_grad = False

    train_data = dataset[train_idx]
    train_labels = labels[train_idx]
    test_data = dataset[test_idx]
    test_labels = labels[test_idx]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd,
                                  betas=(0.9, 0.98))

    checkpoints, checkpoint_epochs = [], []
    train_losses, test_losses = [], []

    for epoch in tqdm(range(n_epochs), desc=f"training"):
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
    }


@torch.no_grad()
def cache_pairs(model, dataset, labels, train_idx, layer, device, n_pairs=200):
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


# ── Grassmannian geometry primitives ─────────────────────────────────

def principal_angles(Q1, Q2):
    """Canonical angles between two subspaces (columns of Q1, Q2).

    Returns angles in radians. The singular values of Q1^T Q2 are cos(theta_i).
    """
    M = Q1.T @ Q2
    svals = torch.linalg.svdvals(M).clamp(-1, 1)
    return torch.acos(svals)


def grassmannian_distance(Q1, Q2):
    """Geodesic distance on Gr(k, d): sqrt(sum(theta_i^2))."""
    angles = principal_angles(Q1, Q2)
    return torch.sqrt((angles ** 2).sum()).item()


def grassmannian_log(Q1, Q2):
    """Log map on Gr(k, d): tangent vector at Q1 pointing toward Q2.

    Returns Delta (d x k) with Q1^T Delta = 0 and ||Delta||_F = geodesic distance.
    """
    M = Q1.T @ Q2
    Qperp = Q2 - Q1 @ M
    try:
        QpMi = Qperp @ torch.linalg.inv(M)
    except Exception:
        QpMi = Qperp @ torch.linalg.pinv(M)
    U, S, Vh = torch.linalg.svd(QpMi, full_matrices=False)
    return U @ torch.diag(torch.arctan(S)) @ Vh


def grassmannian_exp(Q, Delta):
    """Exp map: follow tangent vector Delta from Q on Gr(k, d)."""
    U, S, Vh = torch.linalg.svd(Delta, full_matrices=False)
    Q_new = Q @ Vh.T @ torch.diag(torch.cos(S)) @ Vh + U @ torch.diag(torch.sin(S)) @ Vh
    return torch.linalg.qr(Q_new).Q[:, :Q.shape[1]]


def tangent_space_embedding(operation_Qs, reference_op=None):
    """Embed all operations in the tangent space at a reference point.

    Returns dict mapping op -> tangent vector (d x k matrix, flattened to vector).
    If reference_op is None, uses the first operation alphabetically.
    """
    ops = sorted(operation_Qs.keys())
    if reference_op is None:
        reference_op = ops[0]
    Q_ref = operation_Qs[reference_op]

    embedding = {}
    for op in ops:
        if op == reference_op:
            embedding[op] = torch.zeros_like(Q_ref)
        else:
            embedding[op] = grassmannian_log(Q_ref, operation_Qs[op])
    return embedding, reference_op


def analogy_cosine(embedding, op_A, op_B, op_C, op_D):
    """Check if the vector from A→B is parallel to C→D in tangent space.

    Returns cosine similarity of (B - A) and (D - C) as flattened vectors.
    """
    vec_AB = (embedding[op_B] - embedding[op_A]).flatten()
    vec_CD = (embedding[op_D] - embedding[op_C]).flatten()
    cos_sim = F.cosine_similarity(vec_AB.unsqueeze(0), vec_CD.unsqueeze(0)).item()
    return cos_sim


def random_tangent_perturbation(Q, epsilon, device):
    """Perturb Q along a random tangent vector of Gr(k, d).

    Tangent space at Q is {Q_perp @ A : A in R^{(d-k) x k}}.
    Returns a new orthonormal Q' at geodesic distance ~epsilon from Q.
    """
    d, k = Q.shape
    Q_perp = torch.linalg.qr(
        torch.cat([Q, torch.randn(d, d - k, device=device)], dim=1)
    ).Q[:, k:]

    A = torch.randn(d - k, k, device=device)
    A = A / A.norm() * epsilon

    Q_new = Q + Q_perp @ A
    Q_new = torch.linalg.qr(Q_new).Q[:, :k]
    return Q_new


# ── Test 1: Principal angles between operations ─────────────────────

def cross_operation_angles(operation_Qs):
    """Compute pairwise principal angles and geodesic distances."""
    ops = sorted(operation_Qs.keys())
    results = {}
    for i, op1 in enumerate(ops):
        for op2 in ops[i + 1:]:
            Q1 = operation_Qs[op1]
            Q2 = operation_Qs[op2]
            angles = principal_angles(Q1, Q2)
            dist = grassmannian_distance(Q1, Q2)
            results[f"{op1}_vs_{op2}"] = {
                "angles_rad": [float(a) for a in angles],
                "angles_deg": [float(a * 180 / math.pi) for a in angles],
                "geodesic_distance": dist,
            }
    return results


# ── Test 2: Grassmannian trajectory during grokking ──────────────────

def trajectory_on_grassmannian(model, cfg, training_data, dataset, labels,
                               train_idx, layer, k, n_steps, device,
                               sample_every=5):
    """Train DAS at checkpoints, track subspace path on Gr(k, d)."""
    checkpoints = training_data["checkpoints"]
    checkpoint_epochs = training_data["checkpoint_epochs"]
    train_losses = training_data["train_losses"]
    test_losses = training_data["test_losses"]

    if not checkpoints:
        return {"error": "no checkpoints"}

    n_ckpts = len(checkpoints)
    indices = list(range(0, n_ckpts, sample_every))
    if indices[-1] != n_ckpts - 1:
        indices.append(n_ckpts - 1)

    site = Site(
        f"blocks.{layer}.hook_resid_post", cfg.d_model,
        lambda act: act[0, -1, :],
        lambda act, iv: torch.cat([act[:, :-1, :], iv.unsqueeze(0).unsqueeze(0)], dim=1),
    )

    Qs = []
    iias = []
    epochs = []
    losses_at_ckpt = []

    for ci in tqdm(indices, desc="trajectory DAS"):
        epoch = checkpoint_epochs[ci]
        model.load_state_dict(checkpoints[ci])
        model.eval()

        cached = cache_pairs(model, dataset, labels, train_idx, layer, device)
        if len(cached) < 4:
            continue

        Q, iia = train_das(model, site, cached, k, n_steps, device)
        Qs.append(Q)
        iias.append(iia)
        epochs.append(epoch)
        losses_at_ckpt.append(test_losses[epoch] if epoch < len(test_losses) else None)

    if len(Qs) < 2:
        return {"error": "too few valid checkpoints"}

    Q_final = Qs[-1]
    distances_to_final = [grassmannian_distance(Q, Q_final) for Q in Qs]
    consecutive_distances = [
        grassmannian_distance(Qs[i], Qs[i + 1]) for i in range(len(Qs) - 1)
    ]

    max_jump_idx = max(range(len(consecutive_distances)),
                       key=lambda i: consecutive_distances[i])

    return {
        "epochs": epochs,
        "iias": iias,
        "test_losses": losses_at_ckpt,
        "distances_to_final": distances_to_final,
        "consecutive_distances": consecutive_distances,
        "max_jump_epoch": epochs[max_jump_idx],
        "max_jump_distance": consecutive_distances[max_jump_idx],
        "total_path_length": sum(consecutive_distances),
    }


# ── Test 3: Basin width on Gr(k, d) ─────────────────────────────────

def basin_width(model, site, Q, cached_eval, device, n_perturbations=20,
                epsilons=None):
    """Measure IIA degradation as Q is perturbed along Grassmannian tangents."""
    if epsilons is None:
        epsilons = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5]

    baseline_iia = eval_iia(model, site, Q, cached_eval)

    results = {"baseline_iia": baseline_iia, "epsilons": []}
    for eps in epsilons:
        iias = []
        dists = []
        for _ in range(n_perturbations):
            Q_pert = random_tangent_perturbation(Q, eps, device)
            iia = eval_iia(model, site, Q_pert, cached_eval)
            dist = grassmannian_distance(Q, Q_pert)
            iias.append(iia)
            dists.append(dist)

        mean_iia = sum(iias) / len(iias)
        mean_dist = sum(dists) / len(dists)
        results["epsilons"].append({
            "epsilon": eps,
            "mean_iia": mean_iia,
            "std_iia": (sum((x - mean_iia) ** 2 for x in iias) / len(iias)) ** 0.5,
            "mean_geodesic_distance": mean_dist,
            "iia_drop": baseline_iia - mean_iia,
        })

    half_iia = baseline_iia / 2
    basin_eps = None
    for entry in results["epsilons"]:
        if entry["mean_iia"] < half_iia:
            basin_eps = entry["mean_geodesic_distance"]
            break
    results["half_iia_geodesic_distance"] = basin_eps

    return results


# ── Main ─────────────────────────────────────────────────────────────

def utc_ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main():
    parser = argparse.ArgumentParser(description="Grassmannian geometry of DAS subspaces")
    parser.add_argument("--operations", default="multiplication,composite_addition",
                        help="Comma-separated operations to analyze")
    parser.add_argument("--p", type=int, default=113)
    parser.add_argument("--n-epochs", type=int, default=None,
                        help="Override per-operation epoch defaults")
    parser.add_argument("--das-steps", type=int, default=400)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="experiments/results")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--test", default="all",
                        choices=["all", "angles", "trajectory", "basin", "analogies"])
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--trajectory-sample-every", type=int, default=5)
    parser.add_argument("--basin-perturbations", type=int, default=20)
    args = parser.parse_args()

    operations = [op.strip() for op in args.operations.split(",")]
    device = args.device
    k = args.k
    das_steps = args.n_steps if args.n_steps is not None else args.das_steps
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_all = args.test == "all"
    layer = 0

    epoch_defaults = {
        "multiplication": 40000, "polynomial": 60000,
        "composite_addition": 15000, "division": 40000,
        "squaring": 60000, "cubing": 60000,
        "max_ab": 25000, "abs_diff": 60000, "sum_of_squares": 30000,
        "power": 80000, "shifted_mult": 40000,
        "min_ab": 60000, "floor_div": 60000, "bitwise_xor": 60000,
        "gcd": 60000, "subtraction": 60000, "affine": 60000,
        "cubic_sum": 60000, "modular_distance": 60000,
    }
    p_defaults = {
        "composite_addition": 91,
    }

    logger.info("[%s] Grassmannian Geometry — operations: %s, k=%d", utc_ts(), operations, k)

    # Train all models and get DAS subspaces
    operation_Qs = {}
    operation_models = {}
    operation_data = {}

    for op in operations:
        p = p_defaults.get(op, args.p)
        n_epochs = args.n_epochs or epoch_defaults.get(op, 25000)

        logger.info("[%s] === Training %s mod %d (%d epochs) ===", utc_ts(), op, p, n_epochs)

        dataset, labels = build_data(op, p, device)
        frac = 0.5 if op in ("squaring", "cubing") else FRAC_TRAIN
        train_idx, test_idx = make_train_test_split(len(dataset), frac)

        model, cfg, training_data = train_grokking_model(
            p, device, n_epochs=n_epochs, checkpoint_every=args.checkpoint_every,
            dataset=dataset, labels=labels, train_idx=train_idx, test_idx=test_idx,
        )

        final_test_loss = training_data["test_losses"][-1]
        grokked = final_test_loss < 0.1
        logger.info("  Grokked: %s (final test loss: %.4f)", grokked, final_test_loss)

        if not grokked:
            logger.warning("  %s did not grok — skipping Grassmannian analysis", op)
            continue

        site = Site(
            f"blocks.{layer}.hook_resid_post", cfg.d_model,
            lambda act: act[0, -1, :],
            lambda act, iv: torch.cat(
                [act[:, :-1, :], iv.unsqueeze(0).unsqueeze(0)], dim=1
            ),
        )

        cached = cache_pairs(model, dataset, labels, train_idx, layer, device)
        Q, iia = train_das(model, site, cached, k, das_steps, device)
        logger.info("  DAS k=%d: IIA=%.3f", k, iia)

        operation_Qs[op] = Q
        operation_models[op] = (model, cfg, site, cached, dataset, labels,
                                train_idx, test_idx, training_data, p)

    all_results = {"operations": list(operation_Qs.keys()), "k": k}

    # ── Test 1: Principal angles ─────────────────────────────────────
    if (run_all or args.test == "angles") and len(operation_Qs) >= 2:
        logger.info("[%s] === Test 1: Principal angles between operations ===", utc_ts())
        angle_results = cross_operation_angles(operation_Qs)
        for pair, data in angle_results.items():
            logger.info("  %s:", pair)
            logger.info("    Angles: %s", ", ".join(f"{a:.1f}°" for a in data["angles_deg"]))
            logger.info("    Geodesic distance: %.4f", data["geodesic_distance"])
        all_results["principal_angles"] = angle_results

    # ── Test 2: Grassmannian trajectory ──────────────────────────────
    if run_all or args.test == "trajectory":
        logger.info("[%s] === Test 2: Grassmannian trajectory during grokking ===", utc_ts())
        trajectory_results = {}
        for op, (model, cfg, site, cached, dataset, labels,
                 train_idx, test_idx, training_data, p) in operation_models.items():
            logger.info("  %s:", op)
            traj = trajectory_on_grassmannian(
                model, cfg, training_data, dataset, labels, train_idx, layer,
                k, das_steps, device, sample_every=args.trajectory_sample_every,
            )
            if "error" in traj:
                logger.warning("    %s", traj["error"])
                continue

            logger.info("    Path length: %.4f", traj["total_path_length"])
            logger.info("    Max jump: epoch %d (distance %.4f)",
                        traj["max_jump_epoch"], traj["max_jump_distance"])
            logger.info("    Distance to final over time:")
            for i, (ep, d2f, iia) in enumerate(zip(
                traj["epochs"], traj["distances_to_final"], traj["iias"]
            )):
                loss_str = f"loss={traj['test_losses'][i]:.4f}" if traj['test_losses'][i] is not None else "loss=N/A"
                logger.info("      Epoch %6d: d_final=%.4f  IIA=%.3f  %s", ep, d2f, iia, loss_str)

            trajectory_results[op] = traj
        all_results["trajectory"] = trajectory_results

    # ── Test 3: Basin width ──────────────────────────────────────────
    if run_all or args.test == "basin":
        logger.info("[%s] === Test 3: Basin width on Gr(%d, %d) ===", utc_ts(), k, 128)
        basin_results = {}
        for op, Q in operation_Qs.items():
            model, cfg, site, cached, _, _, _, _, _, _ = operation_models[op]
            # Reload final model state
            training_data = operation_models[op][8]
            if training_data["checkpoints"]:
                model.load_state_dict(training_data["checkpoints"][-1])

            n_train = int(len(cached) * 0.75)
            cached_eval = cached[n_train:]

            logger.info("  %s (baseline IIA=%.3f):", op, eval_iia(model, site, Q, cached_eval))
            bw = basin_width(model, site, Q, cached_eval, device,
                             n_perturbations=args.basin_perturbations)

            for entry in bw["epsilons"]:
                logger.info("    eps=%.3f  d_Gr=%.4f  IIA=%.3f ± %.3f  (drop=%.3f)",
                            entry["epsilon"], entry["mean_geodesic_distance"],
                            entry["mean_iia"], entry["std_iia"], entry["iia_drop"])

            if bw["half_iia_geodesic_distance"] is not None:
                logger.info("    Half-IIA geodesic distance: %.4f",
                            bw["half_iia_geodesic_distance"])
            else:
                logger.info("    IIA never dropped below half — very wide basin")

            basin_results[op] = bw
        all_results["basin_width"] = basin_results

    # ── Test 4: Tangent space embedding and analogies ─────────────────
    if (run_all or args.test == "analogies") and len(operation_Qs) >= 2:
        logger.info("[%s] === Test 4: Tangent space embedding and analogies ===", utc_ts())
        embedding, ref_op = tangent_space_embedding(operation_Qs)
        logger.info("  Reference point: %s", ref_op)

        tangent_norms = {}
        for op, delta in embedding.items():
            norm = delta.norm().item()
            tangent_norms[op] = norm
            logger.info("  %s: tangent norm = %.4f (geodesic distance from %s)", op, norm, ref_op)

        pairwise_cosines = {}
        ops = sorted(operation_Qs.keys())
        if len(ops) >= 3:
            for i, op_i in enumerate(ops):
                for op_j in ops[i + 1:]:
                    if op_i == ref_op or op_j == ref_op:
                        continue
                    vec_i = embedding[op_i].flatten()
                    vec_j = embedding[op_j].flatten()
                    if vec_i.norm() > 1e-8 and vec_j.norm() > 1e-8:
                        cs = F.cosine_similarity(vec_i.unsqueeze(0), vec_j.unsqueeze(0)).item()
                        pairwise_cosines[f"{op_i}_vs_{op_j}"] = cs
                        logger.info("  Cosine(%s, %s) in tangent space: %.4f", op_i, op_j, cs)

        analogy_results = []
        known_pairs = [
            ("multiplication", "division", "inverse operation"),
            ("multiplication", "shifted_mult", "conjugation by translation"),
        ]
        for op_A, op_B, relationship in known_pairs:
            if op_A in operation_Qs and op_B in operation_Qs:
                for op_C in ops:
                    if op_C in (op_A, op_B):
                        continue
                    for op_D in ops:
                        if op_D in (op_A, op_B, op_C):
                            continue
                        cos = analogy_cosine(embedding, op_A, op_B, op_C, op_D)
                        if abs(cos) > 0.5:
                            analogy_results.append({
                                "A": op_A, "B": op_B, "C": op_C, "D": op_D,
                                "relationship": relationship,
                                "cosine": cos,
                            })
                            logger.info("  Analogy: %s→%s ≈ %s→%s (cos=%.3f, '%s')",
                                        op_A, op_B, op_C, op_D, cos, relationship)

        all_results["tangent_embedding"] = {
            "reference_op": ref_op,
            "tangent_norms": tangent_norms,
            "pairwise_cosines": pairwise_cosines,
            "analogies": analogy_results,
        }

    # ── Summary ──────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print(f"Grassmannian Geometry — k={k}, operations: {list(operation_Qs.keys())}")
    print()

    if "principal_angles" in all_results:
        print("1. Principal angles between operations:")
        for pair, data in all_results["principal_angles"].items():
            print(f"   {pair}: {', '.join(f'{a:.1f}°' for a in data['angles_deg'])}  "
                  f"(geodesic={data['geodesic_distance']:.4f})")

    if "trajectory" in all_results:
        print()
        print("2. Grassmannian trajectory:")
        for op, traj in all_results["trajectory"].items():
            print(f"   {op}: path_length={traj['total_path_length']:.4f}, "
                  f"max_jump at epoch {traj['max_jump_epoch']} "
                  f"(d={traj['max_jump_distance']:.4f})")

    if "basin_width" in all_results:
        print()
        print("3. Basin width:")
        for op, bw in all_results["basin_width"].items():
            half_d = bw["half_iia_geodesic_distance"]
            print(f"   {op}: baseline_IIA={bw['baseline_iia']:.3f}, "
                  f"half-IIA distance={'%.4f' % half_d if half_d else '>1.5'}")

    if "tangent_embedding" in all_results:
        print()
        print("4. Tangent space embedding (reference: %s):" % all_results["tangent_embedding"]["reference_op"])
        for op, norm in all_results["tangent_embedding"]["tangent_norms"].items():
            print(f"   {op}: d_Gr = {norm:.4f}")
        if all_results["tangent_embedding"]["pairwise_cosines"]:
            print("   Pairwise cosines in tangent space:")
            for pair, cs in all_results["tangent_embedding"]["pairwise_cosines"].items():
                print(f"     {pair}: {cs:.4f}")
        if all_results["tangent_embedding"]["analogies"]:
            print("   Detected analogies (|cos| > 0.5):")
            for a in all_results["tangent_embedding"]["analogies"]:
                print(f"     {a['A']}→{a['B']} ≈ {a['C']}→{a['D']} (cos={a['cosine']:.3f})")

    print("=" * 70)

    # ── Diagnosis ────────────────────────────────────────────────────
    if "principal_angles" in all_results:
        all_far = all(d["geodesic_distance"] > 0.3
                      for d in all_results["principal_angles"].values())
        if all_far:
            print("DIAGNOSIS: GEOMETRICALLY DISTINCT — operations occupy different points on Gr(k,d)")
        else:
            print("DIAGNOSIS: PARTIALLY OVERLAPPING — some operation subspaces are nearby on Gr(k,d)")

    # Save results
    out_path = output_dir / "grassmannian_geometry.jsonl"
    with open(out_path, "a") as f:
        f.write(json.dumps(all_results, default=str) + "\n")
    logger.info("[%s] Results appended to %s", utc_ts(), out_path)


if __name__ == "__main__":
    main()
