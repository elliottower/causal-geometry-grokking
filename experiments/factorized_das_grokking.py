"""Factorized DAS on grokking models: distill into W=S*F, then find causal factors.

Step 1: Train a standard grokking model.
Step 2: Learn a shared factor bank F and sparse selectors S by fitting W = S*F
        to the model's weight matrices (W_Q, W_K, W_V, W_O, W_in, W_out).
Step 3: Run factorized DAS: constrain DAS to the factor span col(F^T),
        with group lasso to find which factors carry the causal variable.
Step 4: Interpret surviving factors via logit lens (decode through W_U).

Usage:
    modal run --detach experiments/factorized_das_grokking.py
    modal run --detach experiments/factorized_das_grokking.py --operations addition,squaring
"""
from __future__ import annotations

import json
import logging
import os
import time
import traceback
from datetime import datetime, timezone

import modal

try:
    import einops
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from tqdm import tqdm
    from transformer_lens import HookedTransformer, HookedTransformerConfig
except (ImportError, AttributeError):
    pass

# -- Modal setup --

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "numpy==1.26.4",
        "setuptools<71",
    )
    .pip_install(
        "transformer-lens==2.11.0",
        "transformers==4.46.3",
        "einops>=0.8",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "tqdm",
    )
)

app = modal.App("factorized-das-grokking", image=image)
results_vol = modal.Volume.from_name("fc-results", create_if_missing=True)

logger = logging.getLogger(__name__)

# -- Constants --

P = 113
FRAC_TRAIN = 0.3
DATA_SEED = 598

OPERATIONS = {
    "addition": {"p": 113},
    "multiplication": {"p": 113},
    "subtraction": {"p": 113},
    "division": {"p": 113},
    "bitwise_xor": {"p": 113},
    "sum_of_squares": {"p": 113},
    "cubic_sum": {"p": 113},
    "cubing": {"p": 113},
    "squaring": {"p": 113},
    "polynomial": {"p": 113},
    "affine": {"p": 113},
    "max_ab": {"p": 113},
    "abs_difference": {"p": 113},
    "power": {"p": 113},
}

EPOCH_DEFAULTS = {
    "addition": 25000,
    "multiplication": 40000,
    "subtraction": 25000,
    "division": 40000,
    "bitwise_xor": 60000,
    "sum_of_squares": 30000,
    "cubic_sum": 60000,
    "cubing": 60000,
    "squaring": 60000,
    "polynomial": 60000,
    "affine": 25000,
    "max_ab": 25000,
    "abs_difference": 60000,
    "power": 80000,
}


def utc_ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ===================================================================
# Pure logic (no Modal dependency)
# ===================================================================

def compute_labels(a_vec, b_vec, operation, p):
    if operation == "addition":
        return (a_vec + b_vec) % p
    elif operation == "multiplication":
        return (a_vec * b_vec) % p
    elif operation == "subtraction":
        return (a_vec - b_vec) % p
    elif operation == "division":
        b_inv = torch.tensor([pow(int(b.item()), p - 2, p) for b in b_vec])
        return ((a_vec * b_inv) % p).long()
    elif operation == "bitwise_xor":
        return (a_vec ^ b_vec) % p
    elif operation == "sum_of_squares":
        return (a_vec * a_vec + b_vec * b_vec) % p
    elif operation == "cubic_sum":
        return (a_vec * a_vec * a_vec + b_vec * b_vec * b_vec) % p
    elif operation == "cubing":
        return (a_vec * a_vec * a_vec) % p
    elif operation == "squaring":
        return (a_vec * a_vec) % p
    elif operation == "polynomial":
        return (a_vec * a_vec + b_vec) % p
    elif operation == "affine":
        return (2 * a_vec + 3 * b_vec + 5) % p
    elif operation == "max_ab":
        return torch.max(a_vec, b_vec) % p
    elif operation == "abs_difference":
        return (a_vec - b_vec).abs() % p
    elif operation == "power":
        return torch.tensor(
            [pow(int(a.item()), int(b.item()), p) for a, b in zip(a_vec, b_vec)]
        ).long()
    raise ValueError(f"Unknown operation: {operation}")


def build_data(operation, p, device):
    is_unary = operation in ("squaring", "cubing")
    excludes_zero = operation in ("multiplication", "division", "power")

    if is_unary:
        a_vals = torch.arange(1, p) if excludes_zero else torch.arange(p)
        b_vals = torch.zeros(len(a_vals), dtype=torch.long)
        a_vec, b_vec = a_vals, b_vals
    elif excludes_zero:
        a_vals = torch.arange(1, p)
        b_vals = torch.arange(1, p)
        a_vec = einops.repeat(a_vals, "i -> (i j)", j=len(b_vals))
        b_vec = einops.repeat(b_vals, "j -> (i j)", i=len(a_vals))
    else:
        a_vec = einops.repeat(torch.arange(p), "i -> (i j)", j=p)
        b_vec = einops.repeat(torch.arange(p), "j -> (i j)", i=p)

    eq_vec = torch.full_like(a_vec, p)
    dataset = torch.stack([a_vec, b_vec, eq_vec], dim=1).to(device)
    labels = compute_labels(a_vec, b_vec, operation, p).to(device)

    n_total = len(dataset)
    torch.manual_seed(DATA_SEED)
    indices = torch.randperm(n_total)
    cutoff = int(n_total * FRAC_TRAIN)
    train_idx = indices[:cutoff]
    test_idx = indices[cutoff:]

    return dataset, labels, train_idx, test_idx, a_vec.to(device), b_vec.to(device)


def train_grokking_model(operation, p, device, n_epochs=25000, lr=1e-3, wd=1.0, seed=999):
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

    dataset, labels, train_idx, test_idx, _, _ = build_data(operation, p, device)
    train_data, train_labels = dataset[train_idx], labels[train_idx]
    test_data, test_labels = dataset[test_idx], labels[test_idx]

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.98),
    )

    for epoch in tqdm(range(n_epochs), desc=f"training {operation}"):
        logits = model(train_data)[:, -1]
        loss = F.cross_entropy(logits, train_labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    with torch.inference_mode():
        test_logits = model(test_data)[:, -1]
        test_loss = F.cross_entropy(test_logits, test_labels).item()
        test_acc = (test_logits.argmax(dim=-1) == test_labels).float().mean().item()

    grokked = test_acc > 0.95
    return model, cfg, test_loss, test_acc, grokked


# ===================================================================
# Step 2: Learn factor bank by fitting W = S * F to model weights
# ===================================================================

def extract_weight_matrices(model):
    """Extract all weight matrices from a 1-layer grokking model.

    Returns list of (name, W) where W has shape (out_dim, d_model) or (d_model, out_dim).
    We normalize everything so factor bank rows are directions in d_model space.
    """
    weights = []
    attn = model.blocks[0].attn

    # W_Q, W_K, W_V: (n_heads, d_model, d_head) -> reshape to (n_heads*d_head, d_model)
    for name, W in [("W_Q", attn.W_Q), ("W_K", attn.W_K), ("W_V", attn.W_V)]:
        W_flat = einops.rearrange(W, "h d_m d_h -> (h d_h) d_m")
        weights.append((name, W_flat.detach()))

    # W_O: (n_heads, d_head, d_model) -> reshape to (n_heads*d_head, d_model)
    W_O = einops.rearrange(attn.W_O, "h d_h d_m -> (h d_h) d_m")
    weights.append(("W_O", W_O.detach()))

    # W_in: (d_model, d_mlp) -> transpose to (d_mlp, d_model)
    mlp = model.blocks[0].mlp
    weights.append(("W_in", mlp.W_in.detach().T))

    # W_out: (d_mlp, d_model) -> already (d_mlp, d_model)
    weights.append(("W_out", mlp.W_out.detach()))

    return weights


def learn_factor_bank(model, n_factors, device, n_steps=2000, lr=1e-3):
    """Learn F (n_factors, d_model) and per-projection S_i so W_i ≈ S_i @ F.

    Minimizes sum_i ||W_i - S_i @ F||^2 over F (shared) and S_i (per-projection).
    """
    weights = extract_weight_matrices(model)
    d_model = weights[0][1].shape[1]

    F_bank = nn.Parameter(torch.randn(n_factors, d_model, device=device) * 0.02)
    selectors = {}
    params = [F_bank]
    for name, W in weights:
        n_channels = W.shape[0]
        S = nn.Parameter(torch.randn(n_channels, n_factors, device=device) * 0.02)
        selectors[name] = S
        params.append(S)

    optimizer = torch.optim.Adam(params, lr=lr)

    target_weights = {name: W.to(device) for name, W in weights}

    losses = []
    for step in tqdm(range(n_steps), desc="learning factor bank"):
        optimizer.zero_grad()

        total_loss = torch.tensor(0.0, device=device)
        for name, W_target in target_weights.items():
            S = selectors[name]
            W_recon = S @ F_bank  # (n_channels, d_model)
            total_loss += F.mse_loss(W_recon, W_target)

        total_loss.backward()
        optimizer.step()

        if (step + 1) % 200 == 0:
            losses.append({"step": step + 1, "loss": total_loss.item()})
            logger.info(f"[{utc_ts()}] Factor bank step {step+1}: loss={total_loss.item():.6f}")

    # Compute per-projection reconstruction error
    recon_errors = {}
    with torch.no_grad():
        for name, W_target in target_weights.items():
            S = selectors[name]
            W_recon = S @ F_bank
            rel_err = (W_recon - W_target).norm() / W_target.norm()
            recon_errors[name] = rel_err.item()

    return F_bank.detach(), {n: s.detach() for n, s in selectors.items()}, recon_errors, losses


# ===================================================================
# Step 3: Factorized DAS with group lasso on factor coefficients
# ===================================================================

def cache_das_pairs(model, dataset, labels, test_idx, device, layer=0, n_pairs=500):
    hook_name = f"blocks.{layer}.hook_resid_post"
    pairs = []
    used = set()
    idx_list = test_idx.tolist()

    for i in range(len(idx_list)):
        if len(pairs) >= n_pairs:
            break
        for j in range(i + 1, len(idx_list)):
            if len(pairs) >= n_pairs:
                break
            bi, si = idx_list[i], idx_list[j]
            if labels[bi] == labels[si]:
                continue
            key = (bi, si)
            if key in used:
                continue
            used.add(key)

            base_toks = dataset[bi:bi + 1]
            source_toks = dataset[si:si + 1]

            with torch.no_grad():
                _, bc = model.run_with_cache(base_toks, names_filter=[hook_name])
                _, sc = model.run_with_cache(source_toks, names_filter=[hook_name])

            pairs.append({
                "base_resid": bc[hook_name][0, -1, :].clone(),
                "source_resid": sc[hook_name][0, -1, :].clone(),
                "base_toks": base_toks,
                "src_id": labels[si].item(),
                "base_id": labels[bi].item(),
            })

    return pairs


def eval_iia(model, data, U, layer, device):
    hook_name = f"blocks.{layer}.hook_resid_post"
    correct, total = 0, 0
    proj = U @ U.T

    for d in data:
        diff = d["source_resid"] - d["base_resid"]
        intervention = proj @ diff

        def make_hook(_interv):
            def hk(act, hook):
                new = act.clone()
                new[0, -1, :] += _interv
                return new
            return hk

        with torch.no_grad():
            logits = model.run_with_hooks(
                d["base_toks"],
                fwd_hooks=[(hook_name, make_hook(intervention))],
            )

        if logits[0, -1, d["src_id"]].item() > logits[0, -1, d["base_id"]].item():
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def train_factorized_das(
    model, data, factor_bank, layer, device,
    k=4, n_steps=200, lr=1e-3, l1_lambda=0.1, reg_type="group_lasso",
):
    """Factorized DAS: U = orth(F^T @ A), with group lasso on rows of A.

    A in R^{n_factors x k}. Each row corresponds to one factor.
    Group lasso ||A_i||_2 per row kills entire factors.
    Surviving factors = the ones that carry the causal variable.
    """
    n_factors, d_model = factor_bank.shape
    hook_name = f"blocks.{layer}.hook_resid_post"

    deltas = torch.stack([d["source_resid"] - d["base_resid"] for d in data])
    _, _, Vh = torch.linalg.svd(deltas, full_matrices=False)
    V_k = Vh[:k].T  # (d_model, k)
    A_init = factor_bank @ V_k  # (n_factors, k)

    A = nn.Parameter(A_init.clone().to(device))
    optimizer = torch.optim.Adam([A], lr=lr)

    history = []
    best_ckpt = {"iia": -1.0, "step": 0, "A": None, "n_active": n_factors}

    micro_batch = 10
    n_train = min(len(data), 100)
    mb_ranges = list(range(0, n_train, micro_batch))

    for step in tqdm(range(n_steps), desc=f"factorized DAS (k={k})", leave=False):
        optimizer.zero_grad()

        for mb_start in mb_ranges:
            U = factor_bank.T @ A  # (d_model, k)
            Q, _ = torch.linalg.qr(U)
            proj = Q @ Q.T

            mb_loss = torch.tensor(0.0, device=device)
            for d in data[mb_start:mb_start + micro_batch]:
                diff = d["source_resid"] - d["base_resid"]
                intervention = proj @ diff

                def make_hook(_interv):
                    def hk(act, hook):
                        new = act.clone()
                        new[0, -1, :] += _interv
                        return new
                    return hk

                logits = model.run_with_hooks(
                    d["base_toks"],
                    fwd_hooks=[(hook_name, make_hook(intervention))],
                )
                log_probs = logits[0, -1, :].log_softmax(dim=-1)
                mb_loss -= log_probs[d["src_id"]]

            scaled = mb_loss / n_train

            if reg_type == "proximal_group_lasso":
                scaled.backward()
            elif reg_type == "group_lasso":
                reg = A.norm(dim=1).sum() * l1_lambda / len(mb_ranges)
                (scaled + reg).backward()
            else:
                reg = A.abs().sum() * l1_lambda / len(mb_ranges)
                (scaled + reg).backward()

        optimizer.step()

        if reg_type == "proximal_group_lasso":
            with torch.no_grad():
                row_norms = A.norm(dim=1, keepdim=True)
                shrink = torch.clamp(1 - (lr * l1_lambda) / row_norms, min=0)
                A.mul_(shrink)

        if (step + 1) % 10 == 0:
            with torch.no_grad():
                U_eval = factor_bank.T @ A
                Q_eval, _ = torch.linalg.qr(U_eval)
                iia = eval_iia(model, data, Q_eval, layer, device)
                row_norms = A.norm(dim=1)
                n_active = (row_norms > row_norms.max() * 0.1).sum().item()
                top_factors = torch.argsort(row_norms, descending=True)[:10].tolist()

                entry = {
                    "step": step + 1, "iia": iia,
                    "n_active": n_active, "top_factors": top_factors,
                }
                history.append(entry)

                if iia > best_ckpt["iia"]:
                    best_ckpt = {
                        "iia": iia, "step": step + 1,
                        "A": A.detach().clone(), "n_active": n_active,
                    }

    A_final = best_ckpt["A"] if best_ckpt["A"] is not None else A.detach()
    U_final = factor_bank.T @ A_final
    Q_final, _ = torch.linalg.qr(U_final)

    row_norms = A_final.norm(dim=1)
    surviving_factors = torch.argsort(row_norms, descending=True)
    n_active = (row_norms > row_norms.max() * 0.1).sum().item()

    return Q_final, A_final, surviving_factors.tolist(), n_active, history, best_ckpt


# ===================================================================
# Step 4: Interpret surviving factors
# ===================================================================

def interpret_factors_logit_lens(factor_bank, surviving_factors, model, n_top=10):
    """Decode top surviving factors through the unembedding matrix (logit lens).

    For each surviving factor f_i (a direction in R^{d_model}), compute
    f_i @ W_U to get logit contributions, then report top tokens.
    """
    W_U = model.unembed.W_U.detach()  # (d_model, d_vocab_out)
    results = {}

    for rank, factor_idx in enumerate(surviving_factors[:n_top]):
        f_i = factor_bank[factor_idx]  # (d_model,)
        logit_contributions = f_i @ W_U  # (d_vocab_out,)

        top_positive = torch.argsort(logit_contributions, descending=True)[:5]
        top_negative = torch.argsort(logit_contributions)[:5]

        results[f"rank_{rank}"] = {
            "factor_idx": int(factor_idx),
            "factor_norm": float(f_i.norm().item()),
            "top_positive_tokens": top_positive.tolist(),
            "top_positive_logits": logit_contributions[top_positive].tolist(),
            "top_negative_tokens": top_negative.tolist(),
            "top_negative_logits": logit_contributions[top_negative].tolist(),
            "logit_std": float(logit_contributions.std().item()),
        }

    return results


def analyze_factor_fourier(factor_bank, surviving_factors, model, p, n_top=10):
    """Check if surviving factors correspond to Fourier frequency components.

    Project each factor through the embedding matrix to see if it selects
    specific Fourier frequencies of the input representation.
    """
    W_E = model.embed.W_E.detach()[:p]  # (p, d_model), exclude eq token

    results = {}
    for rank, factor_idx in enumerate(surviving_factors[:n_top]):
        f_i = factor_bank[factor_idx]  # (d_model,)
        projections = W_E @ f_i  # (p,) — how each input number activates this factor

        fft_result = torch.fft.fft(projections)
        power = fft_result.abs()[:p // 2 + 1]

        dominant_freq = int(torch.argmax(power[1:]).item()) + 1
        total_power = float(power.sum().item())
        dominant_power = float(power[dominant_freq].item())

        results[f"rank_{rank}"] = {
            "factor_idx": int(factor_idx),
            "dominant_freq": dominant_freq,
            "dominant_power": dominant_power,
            "total_power": total_power,
            "concentration": dominant_power / (total_power + 1e-8),
            "top3_freqs": torch.argsort(power[1:], descending=True)[:3].add(1).tolist(),
        }

    return results


def cayley_rotate_factors(factor_bank, model, n_steps=500, lr=5e-4, top_k_logits=100):
    """Cayley rotation: find R (orthogonal) s.t. F' = R @ F minimizes logit-lens entropy.

    Uses the Cayley map: R = (I - A)(I + A)^{-1} where A = triu(A_raw, 1) - triu(A_raw, 1)^T
    (skew-symmetric). Free parameters = upper triangle entries. R stays orthogonal throughout.

    After rotation, ~50% of factors become "peaky" (clear token meaning via logit lens),
    ~50% stay non-peaky (positional/structural). This is a free reparameterization —
    model predictions are unchanged.

    Adapted from factorization-circuits/lib/.../rotate_for_peakiness.py
    """
    n_factors, d_model = factor_bank.shape
    W_U = model.unembed.W_U.detach()  # (d_model, d_vocab_out)
    device = factor_bank.device
    FW = factor_bank @ W_U  # precompute (n_factors, d_vocab_out)

    A_skew = torch.zeros(n_factors, n_factors, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([A_skew], lr=lr)
    I = torch.eye(n_factors, device=device)

    best_entropy = float("inf")
    best_A = A_skew.detach().clone()
    history = []

    for step in tqdm(range(n_steps), desc="Cayley rotation"):
        optimizer.zero_grad()

        A = A_skew.triu(1)
        A = A - A.T
        R = torch.linalg.solve(I + A, I - A)

        rotated_logits = R @ FW  # (n_factors, d_vocab_out)
        top_vals, _ = rotated_logits.topk(top_k_logits, dim=1)
        probs = top_vals.softmax(dim=1)
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=1)
        loss = entropy.mean()

        loss.backward()
        optimizer.step()

        if (step + 1) % 100 == 0:
            with torch.no_grad():
                e = loss.item()
                n_peaky = int((entropy < 3.0).sum())
                n_very_peaky = int((entropy < 2.0).sum())
                history.append({
                    "step": step + 1,
                    "mean_entropy": e,
                    "n_peaky_lt3": n_peaky,
                    "n_very_peaky_lt2": n_very_peaky,
                })
                if e < best_entropy:
                    best_entropy = e
                    best_A = A_skew.detach().clone()
            logger.info(
                f"[{utc_ts()}] Cayley step {step+1}: entropy={e:.4f}, "
                f"peaky(<3)={n_peaky}/{n_factors}, very(<2)={n_very_peaky}/{n_factors}"
            )

    with torch.no_grad():
        A = best_A.triu(1)
        A = A - A.T
        R_best = torch.linalg.solve(I + A, I - A)
        F_rotated = R_best @ factor_bank

        logits = F_rotated @ W_U
        top_vals, _ = logits.topk(top_k_logits, dim=1)
        probs = top_vals.softmax(dim=1)
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=1)
        peaky_mask = entropy < 3.0

    peaky_factors = torch.where(peaky_mask)[0].tolist()
    nonpeaky_factors = torch.where(~peaky_mask)[0].tolist()

    factor_interpretations = {}
    for fi in peaky_factors[:20]:
        factor_logits = F_rotated[fi] @ W_U
        top_pos = torch.argsort(factor_logits, descending=True)[:5]
        factor_interpretations[int(fi)] = {
            "entropy": float(entropy[fi].item()),
            "top_tokens": top_pos.tolist(),
            "top_logits": factor_logits[top_pos].tolist(),
            "is_peaky": True,
        }

    return F_rotated, R_best, {
        "n_peaky": len(peaky_factors),
        "n_nonpeaky": len(nonpeaky_factors),
        "peaky_factors": peaky_factors[:50],
        "nonpeaky_factors": nonpeaky_factors[:50],
        "factor_interpretations": factor_interpretations,
        "history": history,
    }


def plot_factor_analysis(A_final, factor_bank, surviving_factors, model, p, operation, grokked, result_dir, cayley_info=None):
    """Plot factor importance, Fourier alignment, and Cayley rotation results."""
    row_norms = A_final.norm(dim=1).cpu().numpy()
    sorted_norms = np.sort(row_norms)[::-1]

    n_cols = 4 if cayley_info else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))

    axes[0].bar(range(min(50, len(sorted_norms))), sorted_norms[:50], color="steelblue", alpha=0.7)
    axes[0].set_xlabel("Factor rank")
    axes[0].set_ylabel("||A_i||")
    threshold = row_norms.max() * 0.1
    n_active = (row_norms > threshold).sum()
    axes[0].axhline(y=threshold, color="red", linestyle="--", alpha=0.5)
    axes[0].set_title(f"{operation}: {n_active} / {len(row_norms)} factors active")

    W_E = model.embed.W_E.detach().cpu()[:p]
    top_factors = surviving_factors[:5]
    for rank, fi in enumerate(top_factors):
        f_i = factor_bank[fi].cpu()
        projections = W_E @ f_i
        fft_power = torch.fft.fft(projections).abs()[:p // 2 + 1].numpy()
        axes[1].plot(fft_power[1:20], label=f"f_{fi}", alpha=0.7)
    axes[1].set_xlabel("Fourier frequency")
    axes[1].set_ylabel("Power")
    axes[1].set_title("Top factors: Fourier spectrum")
    axes[1].legend(fontsize=7)

    if len(top_factors) >= 2:
        f0 = factor_bank[top_factors[0]].cpu()
        f1 = factor_bank[top_factors[1]].cpu()
        proj0 = (W_E @ f0).numpy()
        proj1 = (W_E @ f1).numpy()
        colors = np.arange(p)
        axes[2].scatter(proj0, proj1, c=colors, cmap="hsv", s=8, alpha=0.7)
        axes[2].set_xlabel(f"Factor {top_factors[0]}")
        axes[2].set_ylabel(f"Factor {top_factors[1]}")
        axes[2].set_title("Embeddings on top 2 factors")
        axes[2].set_aspect("equal")

    if cayley_info and n_cols == 4:
        hist = cayley_info.get("history", [])
        if hist:
            steps = [h["step"] for h in hist]
            entropies = [h["mean_entropy"] for h in hist]
            axes[3].plot(steps, entropies, "o-", color="purple")
            axes[3].set_xlabel("Cayley step")
            axes[3].set_ylabel("Mean entropy")
            n_peaky = cayley_info.get("n_peaky", 0)
            n_total = n_peaky + cayley_info.get("n_nonpeaky", 0)
            axes[3].set_title(f"Cayley: {n_peaky}/{n_total} peaky")

    plt.suptitle(f"{operation} ({'grokked' if grokked else 'NOT grokked'})", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(result_dir, f"factor_analysis_{operation}.png"), dpi=150)
    plt.close(fig)


# ===================================================================
# Main pipeline
# ===================================================================

def _save_incremental(result, save_path, vol=None):
    if save_path is None:
        return
    with open(save_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    if vol is not None:
        vol.commit()
    logger.info(f"[{utc_ts()}] Incremental save -> {save_path}")


def run_single_operation(operation, device="cuda", n_factors=256, save_path=None, vol=None):
    p = OPERATIONS[operation]["p"]
    n_epochs = EPOCH_DEFAULTS.get(operation, 25000)

    logger.info(f"[{utc_ts()}] === {operation} (p={p}) ===")

    logger.info(f"[{utc_ts()}] Step 1: Training grokking model ({n_epochs} epochs)...")
    model, cfg, test_loss, test_acc, grokked = train_grokking_model(
        operation, p, device, n_epochs=n_epochs,
    )
    logger.info(f"[{utc_ts()}] Trained: loss={test_loss:.4f}, acc={test_acc:.4f}, grokked={grokked}")

    logger.info(f"[{utc_ts()}] Step 2: Learning factor bank (n_factors={n_factors})...")
    factor_bank, selectors, recon_errors, fb_losses = learn_factor_bank(
        model, n_factors=n_factors, device=device, n_steps=3000, lr=1e-3,
    )
    logger.info(f"[{utc_ts()}] Factor bank learned. Recon errors: {recon_errors}")

    dataset, labels, train_idx, test_idx, a_vec, b_vec = build_data(operation, p, device)
    logger.info(f"[{utc_ts()}] Step 3: Caching DAS pairs...")
    pairs = cache_das_pairs(model, dataset, labels, test_idx, device, n_pairs=500)
    logger.info(f"[{utc_ts()}] Cached {len(pairs)} pairs")

    result = {
        "operation": operation,
        "p": p,
        "n_factors": n_factors,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "grokked": grokked,
        "recon_errors": recon_errors,
        "factorized_das_results": {},
        "vanilla_das_results": {},
    }
    _save_incremental(result, save_path, vol)

    for k in [2, 4]:
        for l1_lam in [0.01, 0.05, 0.1, 0.5]:
            run_key = f"k{k}_lam{l1_lam}"
            logger.info(f"[{utc_ts()}] Factorized DAS k={k}, lambda={l1_lam}...")

            Q, A_final, surviving, n_active, history, best_ckpt = train_factorized_das(
                model, pairs, factor_bank, layer=0, device=device,
                k=k, n_steps=200, lr=1e-3,
                l1_lambda=l1_lam, reg_type="group_lasso",
            )

            iia_final = eval_iia(model, pairs, Q, 0, device)
            logit_lens = interpret_factors_logit_lens(factor_bank, surviving, model, n_top=5)
            fourier_info = analyze_factor_fourier(factor_bank, surviving, model, p, n_top=5)

            logger.info(
                f"[{utc_ts()}]   IIA={iia_final:.4f}, n_active={n_active}, "
                f"top_factor={surviving[0]}"
            )

            result["factorized_das_results"][run_key] = {
                "k": k,
                "l1_lambda": l1_lam,
                "iia": iia_final,
                "best_iia": best_ckpt["iia"],
                "n_active_factors": n_active,
                "surviving_factors_top20": surviving[:20],
                "logit_lens": logit_lens,
                "fourier_analysis": fourier_info,
                "history": history,
            }
            _save_incremental(result, save_path, vol)

    for k in [2, 4]:
        logger.info(f"[{utc_ts()}] Vanilla DAS k={k} (baseline)...")
        d_model = pairs[0]["base_resid"].shape[0]
        identity_bank = torch.eye(d_model, device=device)

        Q_v, A_v, _, _, _, best_v = train_factorized_das(
            model, pairs, identity_bank, layer=0, device=device,
            k=k, n_steps=200, lr=1e-3,
            l1_lambda=0.0, reg_type="l1",
        )
        iia_v = eval_iia(model, pairs, Q_v, 0, device)
        result["vanilla_das_results"][f"k{k}"] = {
            "k": k, "iia": iia_v, "best_iia": best_v["iia"],
        }
        _save_incremental(result, save_path, vol)

    logger.info(f"[{utc_ts()}] Step 4: Cayley rotation for interpretable factors...")
    F_rotated, R_cayley, cayley_info = cayley_rotate_factors(
        factor_bank, model, n_steps=500, lr=5e-4,
    )
    result["cayley_rotation"] = cayley_info
    _save_incremental(result, save_path, vol)
    logger.info(
        f"[{utc_ts()}] Cayley done: {cayley_info['n_peaky']} peaky / "
        f"{cayley_info['n_peaky'] + cayley_info['n_nonpeaky']} total"
    )

    logger.info(f"[{utc_ts()}] Step 5: Factorized DAS with Cayley-rotated factors...")
    Q_rot, A_rot, surviving_rot, n_active_rot, hist_rot, best_rot = train_factorized_das(
        model, pairs, F_rotated, layer=0, device=device,
        k=2, n_steps=200, lr=1e-3,
        l1_lambda=0.05, reg_type="group_lasso",
    )
    iia_rot = eval_iia(model, pairs, Q_rot, 0, device)
    logit_lens_rot = interpret_factors_logit_lens(F_rotated, surviving_rot, model, n_top=10)

    result["cayley_das_results"] = {
        "iia": iia_rot,
        "best_iia": best_rot["iia"],
        "n_active_factors": n_active_rot,
        "surviving_factors_top20": surviving_rot[:20],
        "logit_lens": logit_lens_rot,
        "history": hist_rot,
    }
    _save_incremental(result, save_path, vol)
    logger.info(f"[{utc_ts()}] Cayley DAS IIA={iia_rot:.4f}, active={n_active_rot}")

    best_A = A_rot
    best_surviving = surviving_rot

    return result, model, factor_bank, best_A, best_surviving, cayley_info


# ===================================================================
# Modal functions
# ===================================================================

@app.function(gpu="A100", timeout=14400, volumes={"/results": results_vol})
def run_operation(operation: str, n_factors: int = 256) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    t0 = time.time()
    result_dir = f"/results/grassmannian_atlas/factorized_das/{operation}"
    os.makedirs(result_dir, exist_ok=True)

    out_path = os.path.join(result_dir, "results.json")

    try:
        result, model, factor_bank, best_A, best_surviving, cayley_info = run_single_operation(
            operation, device="cuda", n_factors=n_factors,
            save_path=out_path, vol=results_vol,
        )
        result["elapsed_seconds"] = round(time.time() - t0, 1)
        result["status"] = "success"

        plot_factor_analysis(
            best_A, factor_bank.cpu(), best_surviving,
            model, OPERATIONS[operation]["p"],
            operation, result["grokked"], result_dir,
            cayley_info=cayley_info,
        )
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"[{utc_ts()}] FAILED: {e}\n{tb}")
        result = {
            "operation": operation, "status": "error",
            "error": str(e), "traceback": tb[-2000:],
            "elapsed_seconds": round(time.time() - t0, 1),
        }

    _save_incremental(result, out_path, results_vol)
    return result


@app.local_entrypoint()
def main(operations: str = "", n_factors: int = 256):
    if operations:
        op_list = [o.strip() for o in operations.split(",")]
    else:
        op_list = list(OPERATIONS.keys())

    print(f"Factorized DAS Grokking -- {len(op_list)} operations, {n_factors} factors")
    print(f"  Operations: {op_list}")
    print(f"  Started: {utc_ts()}")
    print()

    t0 = time.time()
    handles = []
    for op in op_list:
        h = run_operation.spawn(operation=op, n_factors=n_factors)
        handles.append((op, h))
        print(f"  Spawned {op} (A100)")

    print(f"\n{len(handles)} containers spawned. Collecting results...\n")

    results = []
    for op, h in handles:
        try:
            result = h.get()
        except Exception as e:
            result = {"operation": op, "status": "error", "error": str(e)}
        results.append(result)

        status = result.get("status", "?")
        elapsed = result.get("elapsed_seconds", 0)
        grok = "Y" if result.get("grokked") else "N"
        best_k2 = result.get("factorized_das_results", {}).get("k2_lam0.05", {})
        van_k2 = result.get("vanilla_das_results", {}).get("k2", {})
        print(f"  {op:20s}  {status:7s}  grok={grok}  {elapsed:6.0f}s  "
              f"fDAS_IIA={best_k2.get('iia', 0):.3f}  "
              f"vanilla_IIA={van_k2.get('iia', 0):.3f}  "
              f"active={best_k2.get('n_active_factors', '?')}")

    total = time.time() - t0
    successes = sum(1 for r in results if r.get("status") == "success")
    print(f"\nFactorized DAS complete: {successes}/{len(results)} in {total:.0f}s")
