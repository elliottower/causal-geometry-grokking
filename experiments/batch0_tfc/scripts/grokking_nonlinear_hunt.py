"""Grokking Nonlinear Hunt — Testing representations beyond modular addition.

Three operations that might produce genuinely nonlinear representations:
1. Modular multiplication (a * b mod P) — multiplicative group structure
2. Modular polynomial ((a² + b) mod P) — quadratic input dependence
3. Composite modular addition ((a + b) mod P, P composite) — product group / torus

Predictions:
  - Multiplication: intrinsic dimension may differ from k=2; additive Fourier R²
    should be LOW (wrong basis for multiplicative group).
  - Polynomial: equivariance under additive shifts should work for the b input,
    but the quadratic a dependence may break clean representation structure.
  - Composite addition (P=91=7×13): CRT decomposition Z_91 ≅ Z_7 × Z_13 predicts
    k=4 intrinsic dimension (2 per factor) and 2 persistent H1 loops (torus).

Usage:
    # Modular multiplication
    python -u experiments/batch6_atlas/grokking_nonlinear_hunt.py \
        --operation multiplication --p 113 --n-epochs 40000 --device cuda

    # Polynomial
    python -u experiments/batch6_atlas/grokking_nonlinear_hunt.py \
        --operation polynomial --p 113 --n-epochs 25000 --device cuda

    # Composite addition
    python -u experiments/batch6_atlas/grokking_nonlinear_hunt.py \
        --operation composite_addition --p 91 --n-epochs 15000 --device cuda

    # Local test
    python -u experiments/batch6_atlas/grokking_nonlinear_hunt.py \
        --operation multiplication --p 17 --n-epochs 50 --das-steps 5 --device cpu
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

FRAC_TRAIN = 0.3
DATA_SEED = 598


def utc_ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Data generation ──

def build_data(operation, p, device, seed=0):
    if operation == "multiplication":
        a_vals = torch.arange(1, p)
        b_vals = torch.arange(1, p)
        a_vec = einops.repeat(a_vals, "i -> (i j)", j=len(b_vals))
        b_vec = einops.repeat(b_vals, "j -> (i j)", i=len(a_vals))
        eq_vec = torch.full_like(a_vec, p)
        dataset = torch.stack([a_vec, b_vec, eq_vec], dim=1).to(device)
        labels = ((a_vec * b_vec) % p).to(device)
    elif operation == "polynomial":
        a_vec = einops.repeat(torch.arange(p), "i -> (i j)", j=p)
        b_vec = einops.repeat(torch.arange(p), "j -> (i j)", i=p)
        eq_vec = torch.full((p * p,), p)
        dataset = torch.stack([a_vec, b_vec, eq_vec], dim=1).to(device)
        labels = ((a_vec ** 2 + b_vec) % p).to(device)
    elif operation == "composite_addition":
        a_vec = einops.repeat(torch.arange(p), "i -> (i j)", j=p)
        b_vec = einops.repeat(torch.arange(p), "j -> (i j)", i=p)
        eq_vec = torch.full((p * p,), p)
        dataset = torch.stack([a_vec, b_vec, eq_vec], dim=1).to(device)
        labels = ((a_vec + b_vec) % p).to(device)
    elif operation == "division":
        a_vals = torch.arange(1, p)
        b_vals = torch.arange(1, p)
        a_vec = einops.repeat(a_vals, "i -> (i j)", j=len(b_vals))
        b_vec = einops.repeat(b_vals, "j -> (i j)", i=len(a_vals))
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
    elif operation == "max_ab":
        # Piecewise linear with fold at a=b. No group action.
        a_vec = einops.repeat(torch.arange(p), "i -> (i j)", j=p)
        b_vec = einops.repeat(torch.arange(p), "j -> (i j)", i=p)
        eq_vec = torch.full((p * p,), p)
        dataset = torch.stack([a_vec, b_vec, eq_vec], dim=1).to(device)
        labels = torch.max(a_vec, b_vec).to(device) % p
    elif operation == "abs_diff":
        # V-shaped fold at a=b. Shift-invariant: |a-b| = |(a+s)-(b+s)|.
        a_vec = einops.repeat(torch.arange(p), "i -> (i j)", j=p)
        b_vec = einops.repeat(torch.arange(p), "j -> (i j)", i=p)
        eq_vec = torch.full((p * p,), p)
        dataset = torch.stack([a_vec, b_vec, eq_vec], dim=1).to(device)
        labels = ((a_vec - b_vec).abs() % p).to(device)
    elif operation == "sum_of_squares":
        # Quadratic, no group action. Level sets are circles in (a,b).
        a_vec = einops.repeat(torch.arange(p), "i -> (i j)", j=p)
        b_vec = einops.repeat(torch.arange(p), "j -> (i j)", i=p)
        eq_vec = torch.full((p * p,), p)
        dataset = torch.stack([a_vec, b_vec, eq_vec], dim=1).to(device)
        labels = ((a_vec * a_vec + b_vec * b_vec) % p).to(device)
    elif operation == "power":
        # a^b mod p. Highly nonlinear — uses discrete exponentiation.
        a_vals = torch.arange(1, p)
        b_vals = torch.arange(1, p)
        a_vec = einops.repeat(a_vals, "i -> (i j)", j=len(b_vals))
        b_vec = einops.repeat(b_vals, "j -> (i j)", i=len(a_vals))
        eq_vec = torch.full_like(a_vec, p)
        dataset = torch.stack([a_vec, b_vec, eq_vec], dim=1).to(device)
        labels = torch.tensor([pow(int(a.item()), int(b.item()), p)
                               for a, b in zip(a_vec, b_vec)]).to(device)
    elif operation == "shifted_mult":
        # (a+1)(b+1)-1 mod p. Isomorphic to multiplication — control.
        a_vec = einops.repeat(torch.arange(p), "i -> (i j)", j=p)
        b_vec = einops.repeat(torch.arange(p), "j -> (i j)", i=p)
        eq_vec = torch.full((p * p,), p)
        dataset = torch.stack([a_vec, b_vec, eq_vec], dim=1).to(device)
        labels = (((a_vec + 1) * (b_vec + 1) - 1) % p).to(device)
    elif operation == "min_ab":
        # min(a, b) mod p. Complement of max — partial additive structure when a <= b.
        a_vec = einops.repeat(torch.arange(p), "i -> (i j)", j=p)
        b_vec = einops.repeat(torch.arange(p), "j -> (i j)", i=p)
        eq_vec = torch.full((p * p,), p)
        dataset = torch.stack([a_vec, b_vec, eq_vec], dim=1).to(device)
        labels = torch.min(a_vec, b_vec).to(device) % p
    elif operation == "floor_div":
        # a // b (integer division, NOT modular inverse). Highly nonlinear — step function.
        a_vals = torch.arange(p)
        b_vals = torch.arange(1, p)
        a_vec = einops.repeat(a_vals, "i -> (i j)", j=len(b_vals))
        b_vec = einops.repeat(b_vals, "j -> (i j)", i=len(a_vals))
        eq_vec = torch.full_like(a_vec, p)
        dataset = torch.stack([a_vec, b_vec, eq_vec], dim=1).to(device)
        labels = (a_vec // b_vec % p).to(device)
    elif operation == "bitwise_xor":
        # a XOR b mod p. Bitwise operation — no algebraic group structure in mod-p.
        a_vec = einops.repeat(torch.arange(p), "i -> (i j)", j=p)
        b_vec = einops.repeat(torch.arange(p), "j -> (i j)", i=p)
        eq_vec = torch.full((p * p,), p)
        dataset = torch.stack([a_vec, b_vec, eq_vec], dim=1).to(device)
        labels = ((a_vec ^ b_vec) % p).to(device)
    elif operation == "gcd":
        # gcd(a, b) mod p. Number-theoretic, no clean group action.
        a_vals = torch.arange(1, p)
        b_vals = torch.arange(1, p)
        a_vec = einops.repeat(a_vals, "i -> (i j)", j=len(b_vals))
        b_vec = einops.repeat(b_vals, "j -> (i j)", i=len(a_vals))
        eq_vec = torch.full_like(a_vec, p)
        dataset = torch.stack([a_vec, b_vec, eq_vec], dim=1).to(device)
        labels = torch.tensor([int(torch.gcd(a, b).item()) % p
                               for a, b in zip(a_vec, b_vec)]).to(device)
    elif operation == "subtraction":
        # (a - b) mod p. Trivial group action — Grassmannian baseline.
        a_vec = einops.repeat(torch.arange(p), "i -> (i j)", j=p)
        b_vec = einops.repeat(torch.arange(p), "j -> (i j)", i=p)
        eq_vec = torch.full((p * p,), p)
        dataset = torch.stack([a_vec, b_vec, eq_vec], dim=1).to(device)
        labels = ((a_vec - b_vec) % p).to(device)
    elif operation == "affine":
        # (2*a + 3*b + 5) mod p. Linear combination — tests if linearity → Grassmannian.
        a_vec = einops.repeat(torch.arange(p), "i -> (i j)", j=p)
        b_vec = einops.repeat(torch.arange(p), "j -> (i j)", i=p)
        eq_vec = torch.full((p * p,), p)
        dataset = torch.stack([a_vec, b_vec, eq_vec], dim=1).to(device)
        labels = ((2 * a_vec + 3 * b_vec + 5) % p).to(device)
    elif operation == "cubic_sum":
        # (a^3 + b^3) mod p. Binary cubic polynomial — compare to sum_of_squares.
        a_vec = einops.repeat(torch.arange(p), "i -> (i j)", j=p)
        b_vec = einops.repeat(torch.arange(p), "j -> (i j)", i=p)
        eq_vec = torch.full((p * p,), p)
        dataset = torch.stack([a_vec, b_vec, eq_vec], dim=1).to(device)
        labels = ((a_vec ** 3 + b_vec ** 3) % p).to(device)
    elif operation == "modular_distance":
        # min(|a-b|, p-|a-b|). Circular distance on Z_p — interesting geometric structure.
        a_vec = einops.repeat(torch.arange(p), "i -> (i j)", j=p)
        b_vec = einops.repeat(torch.arange(p), "j -> (i j)", i=p)
        eq_vec = torch.full((p * p,), p)
        dataset = torch.stack([a_vec, b_vec, eq_vec], dim=1).to(device)
        raw_diff = (a_vec - b_vec).abs()
        labels = torch.min(raw_diff, p - raw_diff).to(device)
    elif operation == "quartic_sum":
        # (a^4 + b^4) mod p. Binary symmetric polynomial — tests if pattern holds beyond cubic.
        a_vec = einops.repeat(torch.arange(p), "i -> (i j)", j=p)
        b_vec = einops.repeat(torch.arange(p), "j -> (i j)", i=p)
        eq_vec = torch.full((p * p,), p)
        dataset = torch.stack([a_vec, b_vec, eq_vec], dim=1).to(device)
        labels = ((a_vec ** 4 + b_vec ** 4) % p).to(device)
    elif operation == "quintic_sum":
        # (a^5 + b^5) mod p. Binary symmetric polynomial — highest degree tested.
        a_vec = einops.repeat(torch.arange(p), "i -> (i j)", j=p)
        b_vec = einops.repeat(torch.arange(p), "j -> (i j)", i=p)
        eq_vec = torch.full((p * p,), p)
        dataset = torch.stack([a_vec, b_vec, eq_vec], dim=1).to(device)
        labels = ((a_vec ** 5 + b_vec ** 5) % p).to(device)
    elif operation == "affine_scaled":
        # (2a+3b+5) mod p, but equivariance tested with correct 5g scaling.
        a_vec = einops.repeat(torch.arange(p), "i -> (i j)", j=p)
        b_vec = einops.repeat(torch.arange(p), "j -> (i j)", i=p)
        eq_vec = torch.full((p * p,), p)
        dataset = torch.stack([a_vec, b_vec, eq_vec], dim=1).to(device)
        labels = ((2 * a_vec + 3 * b_vec + 5) % p).to(device)
    else:
        raise ValueError(f"Unknown operation: {operation}")

    frac_train = 0.5 if operation in ("squaring", "cubing") else FRAC_TRAIN
    # Use a dedicated generator so the split seed doesn't disturb global RNG state.
    # Combine DATA_SEED with the caller's seed for reproducible but varied splits.
    split_rng = torch.Generator()
    split_rng.manual_seed(DATA_SEED + seed)
    n = len(dataset)
    indices = torch.randperm(n, generator=split_rng)
    cutoff = int(n * frac_train)
    return dataset, labels, indices[:cutoff], indices[cutoff:]


def predict_key_freqs(operation, p):
    """Predict which Fourier frequencies should be active based on the operation."""
    if operation == "composite_addition":
        freqs = set()
        for d in range(2, p):
            if p % d == 0:
                freqs.add(d)
                freqs.add(p // d)
                for mult in range(2, p // 2 + 1):
                    if d * mult <= p // 2:
                        freqs.add(d * mult)
                    if (p // d) * mult <= p // 2:
                        freqs.add((p // d) * mult)
        return sorted(f for f in freqs if f <= p // 2)[:8]
    return []


# ── Training ──

def train_grokking_model(p, device, n_epochs=25000, checkpoint_every=500,
                         lr=1e-3, wd=1.0, dataset=None, labels=None,
                         train_idx=None, test_idx=None, seed=999):
    cfg = HookedTransformerConfig(
        n_layers=1, n_heads=4, d_model=128, d_head=32, d_mlp=512,
        act_fn="relu", normalization_type=None,
        d_vocab=p + 1, d_vocab_out=p, n_ctx=3,
        init_weights=True, device=device, seed=seed,
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
    }


# ── DAS and activation caching ──

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


@torch.no_grad()
def get_all_activations(model, dataset, layer, device):
    hook_name = f"blocks.{layer}.hook_resid_post"
    batch_size = 256
    all_acts = []
    for start in range(0, len(dataset), batch_size):
        batch = dataset[start:start + batch_size]
        _, cache = model.run_with_cache(batch, names_filter=[hook_name])
        all_acts.append(cache[hook_name][:, -1, :])
    return torch.cat(all_acts, dim=0)


# ── Geometry tests ──

@torch.no_grad()
def circular_r2_per_frequency(activations, Q, p, freqs=None):
    if freqs is None:
        freqs = list(range(1, p // 2 + 1))

    projected = activations @ Q
    proj_cube = projected.reshape(p, p, -1) if activations.shape[0] == p * p else None

    if proj_cube is None:
        n_side = int(activations.shape[0] ** 0.5)
        if n_side * n_side != activations.shape[0]:
            return {f: 0.0 for f in freqs}
        proj_cube = projected.reshape(n_side, n_side, -1)

    n_a = proj_cube.shape[0]
    results = {}
    for freq in freqs:
        a_idx = torch.arange(n_a, device=activations.device).float()
        cos_a = torch.cos(2 * torch.pi * freq * a_idx / p)
        sin_a = torch.sin(2 * torch.pi * freq * a_idx / p)

        mean_proj = proj_cube.mean(dim=1)
        total_var = ((mean_proj - mean_proj.mean(dim=0, keepdim=True)) ** 2).sum()
        if total_var < 1e-10:
            results[freq] = 0.0
            continue

        cos_proj = (cos_a[:, None] * mean_proj).sum(dim=0)
        sin_proj = (sin_a[:, None] * mean_proj).sum(dim=0)
        explained = (cos_proj ** 2 + sin_proj ** 2).sum() / (n_a / 2)
        r2 = float((explained / total_var).clamp(0, 1).item())
        results[freq] = r2

    return results


@torch.no_grad()
def equivariance_test(model, site, Q, dataset, labels, device, p,
                      operation="multiplication", n_test=200,
                      shift_type="auto"):
    """Test equivariance under additive or multiplicative shifts.

    shift_type: "auto" (infer from operation), "additive", "multiplicative",
                or "scaled_additive:N" (additive with output scaling factor N).
    """
    hook_name = site.hook_name
    proj = Q @ Q.T

    rng = torch.Generator(device="cpu")
    rng.manual_seed(42)
    test_indices = torch.randperm(len(dataset), generator=rng)[:n_test]

    equivariant_count = 0
    total_tested = 0

    scale_factor = 1
    multiplicative_ops = ("multiplication", "division", "shifted_mult", "power")
    if shift_type.startswith("scaled_additive:"):
        use_multiplicative = False
        scale_factor = int(shift_type.split(":")[1])
        shift_type_label = f"scaled_additive({scale_factor})"
    elif shift_type == "auto":
        use_multiplicative = operation in multiplicative_ops
        if operation == "affine_scaled":
            use_multiplicative = False
            scale_factor = 5
        shift_type_label = "multiplicative" if use_multiplicative else "additive"
    else:
        use_multiplicative = shift_type == "multiplicative"
        shift_type_label = shift_type

    if use_multiplicative:
        shifts = [2, 3, 5]
    else:
        shifts = [1, 2, 5]

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

        for shift in shifts:
            if use_multiplicative:
                target_answer = (true_label * shift) % p
                if target_answer == 0 or target_answer == true_label:
                    continue
            else:
                target_answer = (true_label + shift * scale_factor) % p

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
        "shift_type": shift_type_label,
    }


def grassmann_distance(Q1, Q2):
    k = min(Q1.shape[1], Q2.shape[1])
    _, S, _ = torch.linalg.svd(Q1[:, :k].T @ Q2[:, :k])
    return float(torch.acos(S.clamp(-1.0, 1.0)).norm().item())


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="Grokking Nonlinear Hunt")
    parser.add_argument("--operation", required=True,
                        choices=["multiplication", "polynomial", "composite_addition",
                                 "division", "squaring", "cubing",
                                 "max_ab", "abs_diff", "sum_of_squares",
                                 "power", "shifted_mult",
                                 "min_ab", "floor_div", "bitwise_xor", "gcd",
                                 "subtraction", "affine", "cubic_sum",
                                 "modular_distance",
                                 "quartic_sum", "quintic_sum", "affine_scaled"])
    parser.add_argument("--p", type=int, default=113)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-epochs", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--das-steps", type=int, default=400)
    parser.add_argument("--output-dir", default="experiments/results")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--model", default="grokking")
    parser.add_argument("--task", default=None)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--das-steps-ablation", action="store_true",
                        help="Run DAS convergence ablation (200, 400, 800, 1200 steps)")
    args = parser.parse_args()

    # Set seed before any model creation or data splitting
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    operation = args.operation
    p = args.p
    device = args.device
    layer = 0
    das_steps = args.n_steps or args.das_steps
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epoch_defaults = {
        "multiplication": 40000, "polynomial": 25000, "composite_addition": 15000,
        "division": 40000, "squaring": 25000, "cubing": 25000,
    }
    n_epochs = args.n_epochs or epoch_defaults.get(operation, 25000)

    key_freqs = predict_key_freqs(operation, p)
    all_freqs = list(range(1, p // 2 + 1))

    logger.info("[%s] Grokking Nonlinear Hunt — %s mod %d", utc_ts(), operation, p)
    logger.info("  Operation: %s, P: %d, epochs: %d, DAS steps: %d, seed: %d",
                operation, p, n_epochs, das_steps, args.seed)
    if key_freqs:
        logger.info("  Predicted key frequencies: %s", key_freqs)

    dataset, labels, train_idx, test_idx = build_data(operation, p, device, seed=args.seed)
    logger.info("  Dataset: %d examples (%d train, %d test)",
                len(dataset), len(train_idx), len(test_idx))

    # Train grokking model
    logger.info("[%s] Training grokking model (%d epochs)...", utc_ts(), n_epochs)
    model, cfg, training_data = train_grokking_model(
        p, device, n_epochs=n_epochs, checkpoint_every=args.checkpoint_every,
        dataset=dataset, labels=labels, train_idx=train_idx, test_idx=test_idx,
        seed=args.seed,
    )
    checkpoints = training_data["checkpoints"]
    checkpoint_epochs = training_data["checkpoint_epochs"]
    train_losses = training_data["train_losses"]
    test_losses = training_data["test_losses"]

    # Check if model grokked
    final_test_loss = test_losses[-1] if test_losses else float('inf')
    grokked = final_test_loss < 0.1
    logger.info("  Final test loss: %.4f — %s", final_test_loss,
                "GROKKED" if grokked else "NOT GROKKED")

    model.eval()
    d = model.cfg.d_model
    site = site_resid(layer, d)

    # === Test 1: k-sweep ===
    logger.info("\n[%s] === k-sweep (intrinsic dimension) ===", utc_ts())
    cached = cache_pairs(model, dataset, labels, train_idx, layer, device)
    k_sweep_results = []
    for k_val in [2, 4, 6, 8, 10, 12, 16, 20, 24, 32]:
        Q, iia = train_das(model, site, cached, k_val, das_steps, device)
        k_sweep_results.append({"k": k_val, "iia": iia, "Q": Q.cpu().tolist()})
        logger.info("  k=%2d: IIA=%.3f", k_val, iia)

    intrinsic_dim = None
    for r in k_sweep_results:
        if r["iia"] > 0.9:
            intrinsic_dim = r["k"]
            break

    # === Test 2: Circular R² ===
    logger.info("\n[%s] === Circular R² per frequency ===", utc_ts())
    Q_final, _ = train_das(model, site, cached, 16, das_steps, device)
    all_acts = get_all_activations(model, dataset, layer, device)

    r2_per_freq = circular_r2_per_frequency(all_acts, Q_final, p, freqs=all_freqs)
    top_freqs = sorted(r2_per_freq.items(), key=lambda x: -x[1])[:10]
    discovered_key_freqs = [f for f, _ in top_freqs[:4]]
    logger.info("  Top 10 frequencies by circular R²:")
    for freq, r2 in top_freqs:
        marker = " ← PREDICTED" if freq in key_freqs else ""
        logger.info("    freq=%3d: R²=%.4f%s", freq, r2, marker)

    mean_top4_r2 = sum(v for _, v in top_freqs[:4]) / 4
    mean_rest_r2 = sum(v for _, v in top_freqs[4:]) / max(1, len(top_freqs) - 4)
    selectivity = mean_top4_r2 / max(mean_rest_r2, 1e-6)
    logger.info("  Mean top-4 R²: %.4f, Mean rest: %.4f, Selectivity: %.1fx",
                mean_top4_r2, mean_rest_r2, selectivity)

    # === Test 3: Equivariance ===
    logger.info("\n[%s] === Equivariance test ===", utc_ts())
    equiv = equivariance_test(model, site, Q_final, dataset, labels, device, p,
                              operation)
    logger.info("  Equivariant (%s): %d/%d = %.3f",
                equiv["shift_type"], equiv["n_equivariant"], equiv["n_tested"],
                equiv["equivariant_fraction"])

    # Run both shift types for operations where both are meaningful
    equiv_both = None
    multiplicative_ops = ("multiplication", "division", "shifted_mult")
    if operation in multiplicative_ops:
        equiv_both = equivariance_test(model, site, Q_final, dataset, labels,
                                       device, p, operation, shift_type="additive")
        logger.info("  Additive equivariance (cross-check): %d/%d = %.3f",
                    equiv_both["n_equivariant"], equiv_both["n_tested"],
                    equiv_both["equivariant_fraction"])

    # === Test 3b: Equivariance controls ===
    logger.info("\n[%s] === Equivariance controls ===", utc_ts())

    rand_equiv_scores = []
    for seed in range(20):
        torch.manual_seed(seed * 1000 + 99)
        Q_rand, _ = torch.linalg.qr(torch.randn(d, 2, device=device))
        eq_r = equivariance_test(model, site, Q_rand, dataset, labels, device, p,
                                 operation, n_test=100)
        rand_equiv_scores.append(eq_r["equivariant_fraction"])
    equiv_random_mean = sum(rand_equiv_scores) / len(rand_equiv_scores)
    equiv_random_std = (sum((x - equiv_random_mean)**2 for x in rand_equiv_scores) / len(rand_equiv_scores)) ** 0.5
    logger.info("  Random k=2 equivariance (20 seeds): %.3f +/- %.3f",
                equiv_random_mean, equiv_random_std)

    if checkpoints:
        mem_epoch = checkpoint_epochs[0]
        model.load_state_dict(checkpoints[0])
        model.eval()
        cached_mem = cache_pairs(model, dataset, labels, train_idx, layer, device)
        if len(cached_mem) >= 10:
            Q_mem, iia_mem = train_das(model, site, cached_mem, 2, min(200, das_steps), device)
            equiv_mem = equivariance_test(model, site, Q_mem, dataset, labels, device, p,
                                          operation)
            logger.info("  Memorization (epoch %d) equivariance: %.3f (IIA=%.3f)",
                        mem_epoch, equiv_mem["equivariant_fraction"], iia_mem)
        else:
            equiv_mem = {"equivariant_fraction": 0.0, "n_tested": 0, "n_equivariant": 0}
            iia_mem = 0.0
            logger.info("  Memorization (epoch %d): too few valid pairs", mem_epoch)
        model.load_state_dict(checkpoints[-1])
        model.eval()
    else:
        mem_epoch = 0
        equiv_mem = {"equivariant_fraction": 0.0, "n_tested": 0, "n_equivariant": 0}
        iia_mem = 0.0
        logger.info("  No checkpoints saved — skipping memorization control")

    # === Test 3c: Circle geometry ===
    logger.info("\n[%s] === Circle geometry ===", utc_ts())
    Q_k2, _ = train_das(model, site, cached, 2, das_steps, device)
    all_acts_2d = all_acts @ Q_k2

    n_labels = p
    centroids = torch.zeros(n_labels, 2, device=device)
    counts = torch.zeros(n_labels, device=device)
    for i in range(len(labels)):
        lbl = labels[i].item()
        if 0 <= lbl < n_labels:
            centroids[lbl] += all_acts_2d[i]
            counts[lbl] += 1
    active_mask = counts > 0
    active_centroids = centroids[active_mask]
    active_counts = counts[active_mask]
    active_centroids = active_centroids / active_counts.unsqueeze(1)

    center = active_centroids.mean(0)
    centered = active_centroids - center
    radii = centered.norm(dim=1)
    radius_cv = (radii.std() / radii.mean()).item() if radii.mean() > 0 else float('inf')

    angles = torch.atan2(centered[:, 1], centered[:, 0])
    angle_diffs = torch.diff(angles)
    angle_diffs = torch.where(angle_diffs > torch.pi, angle_diffs - 2 * torch.pi, angle_diffs)
    angle_diffs = torch.where(angle_diffs < -torch.pi, angle_diffs + 2 * torch.pi, angle_diffs)
    closing = angles[0] - angles[-1]
    if closing > torch.pi:
        closing = closing - 2 * torch.pi
    if closing < -torch.pi:
        closing = closing + 2 * torch.pi
    total_angle = angle_diffs.sum() + closing
    winding_number = (total_angle / (2 * torch.pi)).round().item()

    sorted_by_angle = angles.argsort()
    consec_diffs = (sorted_by_angle[1:] - sorted_by_angle[:-1]) % len(active_centroids)
    modal_step = consec_diffs.mode().values.item()
    ordered_fraction = (consec_diffs == modal_step).float().mean().item()

    circle_result = {
        "radius_cv": radius_cv,
        "winding_number": winding_number,
        "angular_ordering_accuracy": ordered_fraction,
        "n_active_labels": int(active_mask.sum().item()),
        "centroids": active_centroids.cpu().tolist(),
        "Q_k2": Q_k2.cpu().tolist(),
        "all_2d": all_acts_2d.cpu().tolist(),
        "all_labels": labels.cpu().tolist(),
    }
    logger.info("  Radius CV: %.4f, Winding: %d, Ordering: %.3f",
                radius_cv, int(winding_number), ordered_fraction)

    # Random control for circle test
    Q_rand_circle, _ = torch.linalg.qr(torch.randn(d, 2, device=device))
    rand_2d = all_acts @ Q_rand_circle
    rand_centroids = torch.zeros(n_labels, 2, device=device)
    for i in range(len(labels)):
        lbl = labels[i].item()
        if 0 <= lbl < n_labels:
            rand_centroids[lbl] += rand_2d[i]
    rand_centroids_active = rand_centroids[active_mask] / active_counts.unsqueeze(1)
    rand_centered = rand_centroids_active - rand_centroids_active.mean(0)
    rand_radii = rand_centered.norm(dim=1)
    rand_cv = (rand_radii.std() / rand_radii.mean()).item() if rand_radii.mean() > 0 else float('inf')
    rand_centroids_list = rand_centroids_active.cpu().tolist()
    logger.info("  Random control: radius_cv=%.4f", rand_cv)

    # H1 persistence
    h1_result = {}
    try:
        from ripser import ripser as ripser_fn
        centroids_np = active_centroids.cpu().numpy()
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
            logger.info("  H1: top_persistence=%.4f, n_significant=%d, n_total=%d",
                        top_persistence, n_significant, len(h1))
        else:
            h1_result = {"top_persistence": 0.0, "n_significant_loops": 0, "n_total_h1": 0}
            logger.info("  H1: no loops detected")
    except ImportError:
        logger.info("  ripser not available — skipping H1")

    # === Test 4: Trajectory ===
    logger.info("\n[%s] === Subspace trajectory ===", utc_ts())
    trajectory = []
    n_ckpts = len(checkpoints)
    if n_ckpts == 0:
        logger.info("  No checkpoints — skipping trajectory")
    sample_indices = list(range(0, n_ckpts, max(1, n_ckpts // 8))) if n_ckpts > 0 else []
    if sample_indices and sample_indices[-1] != n_ckpts - 1:
        sample_indices.append(n_ckpts - 1)

    for ckpt_idx in tqdm(sample_indices, desc="trajectory"):
        epoch = checkpoint_epochs[ckpt_idx]
        model.load_state_dict(checkpoints[ckpt_idx])
        model.eval()

        cached_ckpt = cache_pairs(model, dataset, labels, train_idx, layer, device,
                                  n_pairs=100)
        if len(cached_ckpt) < 10:
            trajectory.append({"epoch": epoch, "iia": 0.0})
            continue

        Q_ckpt, iia_ckpt = train_das(model, site, cached_ckpt, 16, min(200, das_steps),
                                     device)
        d_grass = grassmann_distance(Q_ckpt, Q_final)

        Q_ckpt_k2, _ = train_das(model, site, cached_ckpt, 2, min(200, das_steps), device)
        equiv_ckpt = equivariance_test(model, site, Q_ckpt_k2, dataset, labels,
                                       device, p, operation, n_test=100)

        test_loss = test_losses[epoch] if epoch < len(test_losses) else None

        trajectory.append({
            "epoch": epoch,
            "test_loss": test_loss,
            "iia": iia_ckpt,
            "grassmann_to_final": d_grass,
            "equivariance": equiv_ckpt["equivariant_fraction"],
        })
        tl_str = f"{test_loss:.4f}" if test_loss is not None else "N/A"
        logger.info("  Epoch %5d: test_loss=%s  IIA=%.3f  d_G=%.3f  equiv=%.3f",
                    epoch, tl_str, iia_ckpt, d_grass,
                    equiv_ckpt["equivariant_fraction"])

    if checkpoints:
        model.load_state_dict(checkpoints[-1])
        model.eval()

    # === Test 5: DAS convergence ablation ===
    das_ablation = {}
    if args.das_steps_ablation:
        logger.info("\n[%s] === DAS convergence ablation ===", utc_ts())
        cached_abl = cache_pairs(model, dataset, labels, train_idx, layer, device)
        for n_s in [200, 400, 800, 1200]:
            Q_abl, iia_abl = train_das(model, site, cached_abl, 2, n_s, device)
            eq_abl = equivariance_test(model, site, Q_abl, dataset, labels,
                                       device, p, operation, n_test=100)
            das_ablation[n_s] = {
                "iia": iia_abl,
                "equivariance": eq_abl["equivariant_fraction"],
            }
            logger.info("  DAS steps=%4d: IIA=%.3f  equiv=%.3f",
                        n_s, iia_abl, eq_abl["equivariant_fraction"])

    # === Assemble results ===
    result = {
        "timestamp": utc_ts(),
        "operation": operation,
        "p": p,
        "seed": args.seed,
        "layer": layer,
        "d_model": d,
        "n_epochs": n_epochs,
        "das_steps": das_steps,
        "grokked": grokked,
        "final_test_loss": final_test_loss,
        "Q_final": Q_final.cpu().tolist(),
        "k_sweep": k_sweep_results,
        "intrinsic_dimension": intrinsic_dim,
        "circular_r2_top10": {str(f): v for f, v in top_freqs},
        "discovered_key_freqs": discovered_key_freqs,
        "predicted_key_freqs": key_freqs,
        "r2_selectivity": selectivity,
        "equivariance": equiv,
        "equivariance_controls": {
            "grokked_das": equiv["equivariant_fraction"],
            "random_k2_mean": equiv_random_mean,
            "random_k2_std": equiv_random_std,
            "memorization_das": equiv_mem["equivariant_fraction"],
            "memorization_epoch": mem_epoch,
            "memorization_iia": iia_mem,
        },
        "circle_geometry": circle_result,
        "circle_random_cv": rand_cv,
        "circle_random_centroids": rand_centroids_list,
        "h1_persistence": h1_result,
        "trajectory": trajectory,
        "cross_shift_equivariance": {
            "additive": equiv_both["equivariant_fraction"] if equiv_both else None,
            "multiplicative": equiv["equivariant_fraction"] if equiv.get("shift_type") == "multiplicative" else None,
        } if equiv_both else None,
        "das_convergence_ablation": das_ablation if das_ablation else None,
    }

    out_file = output_dir / f"grokking_{operation}.jsonl"
    with open(out_file, "a") as f:
        f.write(json.dumps(result, default=str) + "\n")
    logger.info("[%s] Results appended to %s", utc_ts(), out_file)

    # === Summary ===
    print(f"\n{'='*70}")
    print(f"Grokking Nonlinear Hunt — {operation} mod {p}")
    print(f"  Grokked: {grokked} (final test loss: {final_test_loss:.4f})")

    print(f"\n1. Intrinsic causal dimension: k={intrinsic_dim}")
    for r in k_sweep_results:
        print(f"   k={r['k']:2d}: IIA={r['iia']:.3f}")

    print(f"\n2. Circular R² (top 4 freqs):")
    for f, r2 in top_freqs[:4]:
        marker = " ← PREDICTED" if f in key_freqs else ""
        print(f"   freq={f:3d}: R²={r2:.4f}{marker}")
    print(f"   Selectivity: {selectivity:.1f}x")

    group_type = "multiplicative" if operation == "multiplication" else "additive"
    print(f"\n3a. Equivariance ({group_type}): {equiv['equivariant_fraction']:.3f} "
          f"({equiv['n_equivariant']}/{equiv['n_tested']})")
    print(f"\n3b. Controls:")
    print(f"                   DAS k=2          Random k=2 (20 seeds)")
    print(f"   Grokked:        {equiv['equivariant_fraction']:.3f}            "
          f"{equiv_random_mean:.3f} +/- {equiv_random_std:.3f}")
    print(f"   Memorizing:     {equiv_mem['equivariant_fraction']:.3f}")

    print(f"\n3c. Circle geometry:")
    print(f"   Radius CV: {radius_cv:.4f} (random: {rand_cv:.4f})")
    print(f"   Winding number: {int(winding_number)}")
    print(f"   Angular ordering: {ordered_fraction:.3f}")
    if h1_result:
        print(f"   H1: top_persistence={h1_result.get('top_persistence', 'N/A')}, "
              f"n_loops={h1_result.get('n_significant_loops', 'N/A')}")

    print(f"\n4. Trajectory:")
    for t in trajectory[::max(1, len(trajectory) // 5)]:
        tl = t.get("test_loss")
        tl_str = f"{tl:.4f}" if tl is not None else "N/A"
        eq = t.get("equivariance", 0)
        d_g = t.get("grassmann_to_final")
        d_g_str = f"{d_g:.3f}" if d_g is not None else "N/A"
        print(f"   Epoch {t['epoch']:6d}: loss={tl_str}  IIA={t['iia']:.3f}  "
              f"d_G={d_g_str}  equiv={eq:.3f}")

    # Diagnosis
    print(f"\n{'='*70}")
    if not grokked:
        print(f"DIAGNOSIS: NOT GROKKED — model did not generalize in {n_epochs} epochs")
    elif equiv["equivariant_fraction"] > 0.5 and equiv_random_mean < 0.1:
        print(f"DIAGNOSIS: CONTROLLED EQUIVARIANCE — genuine {group_type} "
              f"group-action structure")
    elif equiv["equivariant_fraction"] > 0.5:
        print(f"DIAGNOSIS: EQUIVARIANT — but random controls also high ({equiv_random_mean:.3f})")
    elif intrinsic_dim and intrinsic_dim > 4:
        print(f"DIAGNOSIS: HIGH-DIMENSIONAL — intrinsic dim {intrinsic_dim} suggests "
              f"complex representation")
    else:
        print(f"DIAGNOSIS: INCONCLUSIVE — equivariance={equiv['equivariant_fraction']:.3f}, "
              f"dim={intrinsic_dim}")


if __name__ == "__main__":
    main()
