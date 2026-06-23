"""Structured VAE Atlas -- encode-swap-decode IIA + equivariance across 14 grokking operations.

Trains a structured/semi-supervised VAE on residual-stream activations from grokking
models for 14 modular arithmetic operations, then evaluates:
  1. Encode-swap-decode IIA (can we swap the causal variable and get source behavior?)
  2. Equivariance (does additive shift a->a+1 produce a consistent rotation in z_causal?)
  3. Linearized subspace extraction (QR of composed encoder Jacobian)

Compared against vanilla DAS and random-subspace baselines for each operation.

Usage (Modal, parallel across all 14 operations):
    modal run --detach experiments/batch6_atlas/06_21_2026_UPDATE/structured_vae_atlas.py

    # Single operation locally:
    modal run --detach experiments/batch6_atlas/06_21_2026_UPDATE/structured_vae_atlas.py \
        --operations addition
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

app = modal.App("structured-vae-atlas", image=image)
results_vol = modal.Volume.from_name("fc-results", create_if_missing=True)

# -- Constants --

P = 113
FRAC_TRAIN = 0.3
DATA_SEED = 598

OPERATIONS = {
    "addition": {"fn_str": "addition", "p": 113},
    "multiplication": {"fn_str": "multiplication", "p": 113},
    "composite_addition": {"fn_str": "composite_addition", "p": 91},
    "subtraction": {"fn_str": "subtraction", "p": 113},
    "division": {"fn_str": "division", "p": 113},
    "bitwise_xor": {"fn_str": "bitwise_xor", "p": 113},
    "sum_of_squares": {"fn_str": "sum_of_squares", "p": 113},
    "cubic_sum": {"fn_str": "cubic_sum", "p": 113},
    "cubing": {"fn_str": "cubing", "p": 113},
    "squaring": {"fn_str": "squaring", "p": 113},
    "polynomial": {"fn_str": "polynomial", "p": 113},
    "affine": {"fn_str": "affine", "p": 113},
    "max_ab": {"fn_str": "max_ab", "p": 113},
    "abs_difference": {"fn_str": "abs_difference", "p": 113},
    "power": {"fn_str": "power", "p": 113},
    "quartic_sum": {"fn_str": "quartic_sum", "p": 113},
    "mixed_product": {"fn_str": "mixed_product", "p": 113},
    "symmetric_power": {"fn_str": "symmetric_power", "p": 113},
    "double_add_mult": {"fn_str": "double_add_mult", "p": 113},
}

# Epoch counts tuned per operation (some need longer training to grok)
EPOCH_DEFAULTS = {
    "addition": 25000,
    "multiplication": 40000,
    "composite_addition": 15000,
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
    "quartic_sum": 60000,
    "mixed_product": 60000,
    "symmetric_power": 80000,
    "double_add_mult": 60000,
}


def utc_ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ===================================================================
# Pure logic (no Modal dependency)
# ===================================================================


def compute_labels(a_vec, b_vec, operation, p):
    """Compute labels for a given operation, handling edge cases."""
    if operation == "addition":
        return (a_vec + b_vec) % p
    elif operation == "multiplication":
        return (a_vec * b_vec) % p
    elif operation == "composite_addition":
        return (a_vec + b_vec) % p
    elif operation == "subtraction":
        return (a_vec - b_vec) % p
    elif operation == "division":
        # b_vec should already exclude 0
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
    elif operation == "quartic_sum":
        return (a_vec.pow(4) + b_vec.pow(4)) % p
    elif operation == "mixed_product":
        return (a_vec * b_vec * (a_vec + b_vec)) % p
    elif operation == "symmetric_power":
        return torch.tensor(
            [pow(int(a.item()), int(b.item()), p) + pow(int(b.item()), int(a.item()), p)
             for a, b in zip(a_vec, b_vec)]
        ).long() % p
    elif operation == "double_add_mult":
        return ((a_vec + b_vec).pow(2) + a_vec * b_vec) % p
    else:
        raise ValueError(f"Unknown operation: {operation}")


def build_data(operation, p, device):
    """Build dataset for a grokking operation."""
    is_unary = operation in ("squaring", "cubing")
    excludes_zero = operation in ("multiplication", "division", "power", "symmetric_power")

    if is_unary:
        a_vals = torch.arange(1, p) if excludes_zero else torch.arange(p)
        b_vals = torch.zeros(len(a_vals), dtype=torch.long)
        a_vec = a_vals
        b_vec = b_vals
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
    """Train a 1-layer transformer on a modular arithmetic operation."""
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

    # Evaluate
    model.eval()
    with torch.inference_mode():
        test_logits = model(test_data)[:, -1]
        test_loss = F.cross_entropy(test_logits, test_labels).item()
        test_acc = (test_logits.argmax(dim=-1) == test_labels).float().mean().item()

    grokked = test_acc > 0.95
    return model, cfg, test_loss, test_acc, grokked


def cache_all_activations(model, dataset, device, layer=0):
    """Cache residual-stream activations at last position for all inputs."""
    hook_name = f"blocks.{layer}.hook_resid_post"
    all_acts = []
    batch_size = 256
    n = len(dataset)
    for i in tqdm(range(0, n, batch_size), desc="caching activations", leave=False):
        batch = dataset[i:i + batch_size]
        with torch.inference_mode():
            _, cache = model.run_with_cache(batch, names_filter=[hook_name])
        acts = cache[hook_name][:, -1, :].clone()  # (batch, d_model)
        all_acts.append(acts)
    return torch.cat(all_acts, dim=0)  # (n_total, d_model)


# -- StructuredVAE --

def build_structured_vae(d_input, z_causal_dim, z_nuisance_dim, hidden_dim, n_classes):
    """Build a structured VAE with separate causal and nuisance latents."""
    class StructuredVAE(nn.Module):
        def __init__(self):
            super().__init__()
            z_dim = z_causal_dim + z_nuisance_dim
            self.z_causal_dim = z_causal_dim
            self.z_nuisance_dim = z_nuisance_dim

            self.enc_trunk = nn.Sequential(
                nn.Linear(d_input, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.enc_causal_mu = nn.Linear(hidden_dim, z_causal_dim)
            self.enc_causal_logvar = nn.Linear(hidden_dim, z_causal_dim)
            self.enc_nuisance_mu = nn.Linear(hidden_dim, z_nuisance_dim)
            self.enc_nuisance_logvar = nn.Linear(hidden_dim, z_nuisance_dim)

            self.decoder = nn.Sequential(
                nn.Linear(z_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, d_input),
            )

            self.classifier = nn.Linear(z_causal_dim, n_classes)

        def encode(self, x):
            h = self.enc_trunk(x)
            z_c_mu = self.enc_causal_mu(h)
            z_c_logvar = self.enc_causal_logvar(h)
            z_n_mu = self.enc_nuisance_mu(h)
            z_n_logvar = self.enc_nuisance_logvar(h)
            return z_c_mu, z_c_logvar, z_n_mu, z_n_logvar

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def decode(self, z):
            return self.decoder(z)

        def forward(self, x):
            z_c_mu, z_c_logvar, z_n_mu, z_n_logvar = self.encode(x)
            z_c = self.reparameterize(z_c_mu, z_c_logvar)
            z_n = self.reparameterize(z_n_mu, z_n_logvar)
            z = torch.cat([z_c, z_n], dim=-1)
            x_recon = self.decode(z)
            class_logits = self.classifier(z_c)
            return x_recon, class_logits, z_c_mu, z_c_logvar, z_n_mu, z_n_logvar

        def get_causal_subspace(self):
            """Extract linearized causal subspace from encoder weights."""
            W1 = self.enc_trunk[0].weight.detach()  # (hidden, d_input)
            W2 = self.enc_trunk[2].weight.detach()  # (hidden, hidden)
            W_mu = self.enc_causal_mu.weight.detach()  # (z_causal, hidden)
            W_composed = W_mu @ W2 @ W1  # (z_causal, d_input)
            Q, _ = torch.linalg.qr(W_composed.T)
            return Q  # (d_input, z_causal) orthonormal

    return StructuredVAE()


def train_vae(vae, activations, labels, n_classes, device,
              n_epochs=300, batch_size=256, lr=1e-3, alpha=10.0):
    """Train the structured VAE with reconstruction + KL + classification losses."""
    vae = vae.to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    n = len(activations)

    for epoch in tqdm(range(n_epochs), desc="training VAE", leave=False):
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            x = activations[idx]
            y = labels[idx]

            x_recon, class_logits, z_c_mu, z_c_lv, z_n_mu, z_n_lv = vae(x)

            # Reconstruction loss
            recon_loss = F.mse_loss(x_recon, x)

            # KL divergence for both causal and nuisance
            kl_c = -0.5 * torch.mean(1 + z_c_lv - z_c_mu.pow(2) - z_c_lv.exp())
            kl_n = -0.5 * torch.mean(1 + z_n_lv - z_n_mu.pow(2) - z_n_lv.exp())

            # Classification loss on causal latent
            cls_loss = F.cross_entropy(class_logits, y)

            loss = recon_loss + kl_c + kl_n + alpha * cls_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

    final_loss = epoch_loss / max(n_batches, 1)
    return vae, final_loss


def eval_vae_classifier(vae, activations, labels, device):
    """Evaluate classifier accuracy on the causal latent."""
    vae.eval()
    with torch.inference_mode():
        z_c_mu, _, _, _ = vae.encode(activations)
        logits = vae.classifier(z_c_mu)
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean().item()
    return acc


def eval_vae_reconstruction(vae, activations, device):
    """Evaluate reconstruction MSE."""
    vae.eval()
    with torch.inference_mode():
        x_recon, _, _, _, _, _ = vae(activations)
        recon_loss = torch.nn.functional.mse_loss(x_recon, activations).item()
    return recon_loss


def eval_encode_swap_decode_iia(vae, model, activations, labels, dataset, device,
                                 test_idx, n_pairs=500, layer=0):
    """Encode-swap-decode IIA: swap causal latent, decode, hook into model.

    For each (base, source) pair with different labels:
    1. Encode both -> (z_c, z_n)
    2. Construct z_intervened = (z_c_source, z_n_base)
    3. Decode -> h_intervened
    4. Hook h_intervened into model at the intervention site
    5. Check if model output matches source label
    """
    vae.eval()
    hook_name = f"blocks.{layer}.hook_resid_post"
    test_acts = activations[test_idx]
    test_labels = labels[test_idx]
    test_data = dataset[test_idx]

    # Build pairs with different labels
    n_test = len(test_idx)
    pairs = []
    for i in range(n_test):
        for j in range(i + 1, min(i + 50, n_test)):
            if test_labels[i] != test_labels[j]:
                pairs.append((i, j))
                if len(pairs) >= n_pairs:
                    break
        if len(pairs) >= n_pairs:
            break

    if len(pairs) == 0:
        return 0.0

    correct = 0
    total = 0

    with torch.inference_mode():
        for base_i, src_i in tqdm(pairs, desc="IIA eval", leave=False):
            base_act = test_acts[base_i].unsqueeze(0)
            src_act = test_acts[src_i].unsqueeze(0)
            src_label = test_labels[src_i].item()

            # Encode
            z_c_base_mu, _, z_n_base_mu, _ = vae.encode(base_act)
            z_c_src_mu, _, _, _ = vae.encode(src_act)

            # Swap: source causal, base nuisance (use means, no sampling)
            z_intervened = torch.cat([z_c_src_mu, z_n_base_mu], dim=-1)
            h_intervened = vae.decode(z_intervened).squeeze(0)  # (d_model,)

            # Hook into model
            base_tokens = test_data[base_i].unsqueeze(0)

            def hook_fn(act, hook=None, iv=h_intervened):
                new = act.clone()
                new[0, -1, :] = iv
                return new

            logits = model.run_with_hooks(
                base_tokens, fwd_hooks=[(hook_name, hook_fn)]
            )[0, -1, :]
            pred = logits.argmax(dim=-1).item()

            if pred == src_label:
                correct += 1
            total += 1

    return correct / max(total, 1)


def eval_encode_swap_decode_iia_hard(vae, model, activations, labels, dataset, device,
                                      test_idx, n_pairs=200, layer=0):
    """IIA on 'hard' examples where base and source labels are maximally different.

    Hard = labels differ by more than P/4 in cyclic distance.
    """
    p = labels.max().item() + 1
    vae.eval()
    hook_name = f"blocks.{layer}.hook_resid_post"
    test_acts = activations[test_idx]
    test_labels = labels[test_idx]
    test_data = dataset[test_idx]

    n_test = len(test_idx)
    pairs = []
    for i in range(n_test):
        for j in range(i + 1, min(i + 100, n_test)):
            li, lj = test_labels[i].item(), test_labels[j].item()
            cyclic_dist = min(abs(li - lj), p - abs(li - lj))
            if cyclic_dist > p // 4:
                pairs.append((i, j))
                if len(pairs) >= n_pairs:
                    break
        if len(pairs) >= n_pairs:
            break

    if len(pairs) == 0:
        return 0.0

    correct = 0
    total = 0

    with torch.inference_mode():
        for base_i, src_i in tqdm(pairs, desc="hard IIA", leave=False):
            base_act = test_acts[base_i].unsqueeze(0)
            src_act = test_acts[src_i].unsqueeze(0)
            src_label = test_labels[src_i].item()

            z_c_base_mu, _, z_n_base_mu, _ = vae.encode(base_act)
            z_c_src_mu, _, _, _ = vae.encode(src_act)

            z_intervened = torch.cat([z_c_src_mu, z_n_base_mu], dim=-1)
            h_intervened = vae.decode(z_intervened).squeeze(0)

            base_tokens = test_data[base_i].unsqueeze(0)

            def hook_fn(act, hook=None, iv=h_intervened):
                new = act.clone()
                new[0, -1, :] = iv
                return new

            logits = model.run_with_hooks(
                base_tokens, fwd_hooks=[(hook_name, hook_fn)]
            )[0, -1, :]
            pred = logits.argmax(dim=-1).item()

            if pred == src_label:
                correct += 1
            total += 1

    return correct / max(total, 1)


def eval_equivariance(vae, activations, labels, a_vec, b_vec, p, test_idx, device):
    """Test equivariance under additive shift a -> a+1 mod p.

    For each test input (a, b):
    1. Get z_causal(a, b) and z_causal((a+1) mod p, b)
    2. Compute delta = z_causal(a+1, b) - z_causal(a, b)
    3. Measure consistency: fraction of inputs where ||delta - mean_delta|| < 0.1 * ||mean_delta||
    """
    vae.eval()

    # Build map from (a, b) -> index for quick lookup
    a_all = a_vec.cpu()
    b_all = b_vec.cpu()
    ab_to_idx = {}
    for idx in range(len(a_all)):
        ab_to_idx[(a_all[idx].item(), b_all[idx].item())] = idx

    # Collect deltas for test set pairs that have a valid shifted counterpart
    deltas = []
    with torch.inference_mode():
        z_c_all, _, _, _ = vae.encode(activations)

        for ti in test_idx:
            ti_val = ti.item()
            a_val = a_all[ti_val].item()
            b_val = b_all[ti_val].item()
            a_shifted = (a_val + 1) % p
            shifted_idx = ab_to_idx.get((a_shifted, b_val))
            if shifted_idx is None:
                continue
            delta = z_c_all[shifted_idx] - z_c_all[ti_val]
            deltas.append(delta)

    if len(deltas) < 10:
        return 0.0

    deltas_tensor = torch.stack(deltas)  # (n, z_causal)
    mean_delta = deltas_tensor.mean(dim=0)
    mean_norm = mean_delta.norm().item()

    if mean_norm < 1e-8:
        return 0.0

    deviations = (deltas_tensor - mean_delta).norm(dim=1)
    threshold = 0.1 * mean_norm
    consistent = (deviations < threshold).float().mean().item()

    return consistent


def train_das_baseline(model, activations, labels, dataset, test_idx, device,
                       k=2, n_steps=400, layer=0, n_pairs=500):
    """Train vanilla DAS baseline for comparison."""
    hook_name = f"blocks.{layer}.hook_resid_post"
    d_model = model.cfg.d_model

    test_acts = activations[test_idx]
    test_labels = labels[test_idx]
    test_data = dataset[test_idx]

    # Build training pairs
    n_test = len(test_idx)
    train_pairs = []
    for i in range(n_test):
        for j in range(i + 1, min(i + 30, n_test)):
            if test_labels[i] != test_labels[j]:
                train_pairs.append((i, j))
                if len(train_pairs) >= n_pairs:
                    break
        if len(train_pairs) >= n_pairs:
            break

    if len(train_pairs) < 10:
        return 0.0, 0.0, None

    n_train = int(len(train_pairs) * 0.75)
    das_train = train_pairs[:n_train]
    das_eval = train_pairs[n_train:]

    # DAS parameter
    R = torch.randn(d_model, k, device=device) * 0.02
    R = torch.nn.Parameter(R)
    optimizer = torch.optim.Adam([R], lr=1e-3)

    for step in tqdm(range(n_steps), desc=f"DAS k={k}", leave=False):
        Q, _ = torch.linalg.qr(R)
        proj = Q @ Q.T

        batch_idx = torch.randint(0, len(das_train), (min(16, len(das_train)),))
        loss = torch.tensor(0.0, device=device)
        for bi in batch_idx:
            base_i, src_i = das_train[bi]
            ba = test_acts[base_i]
            sa = test_acts[src_i]
            si = test_labels[src_i].item()

            iv = ba - ba @ proj + sa @ proj
            base_tokens = test_data[base_i].unsqueeze(0)

            def hook_fn(act, hook=None, iv_vec=iv):
                new = act.clone()
                new[0, -1, :] = iv_vec
                return new

            logits = model.run_with_hooks(
                base_tokens, fwd_hooks=[(hook_name, hook_fn)]
            )[0, -1, :]
            log_probs = F.log_softmax(logits, dim=-1)
            loss = loss - log_probs[si]

        loss = loss / len(batch_idx)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Eval IIA
    with torch.no_grad():
        Q, _ = torch.linalg.qr(R)
        Q = Q.detach()
    proj = Q @ Q.T

    correct = 0
    total = 0
    with torch.inference_mode():
        for base_i, src_i in das_eval:
            ba = test_acts[base_i]
            sa = test_acts[src_i]
            si = test_labels[src_i].item()

            iv = ba - ba @ proj + sa @ proj
            base_tokens = test_data[base_i].unsqueeze(0)

            def hook_fn(act, hook=None, iv_vec=iv):
                new = act.clone()
                new[0, -1, :] = iv_vec
                return new

            logits = model.run_with_hooks(
                base_tokens, fwd_hooks=[(hook_name, hook_fn)]
            )[0, -1, :]
            pred = logits.argmax(dim=-1).item()
            if pred == si:
                correct += 1
            total += 1

    iia = correct / max(total, 1)
    return iia, 0.0, Q  # equivariance not computed for DAS baseline here


def eval_das_equivariance(Q, activations, a_vec, b_vec, p, test_idx, device):
    """Evaluate equivariance of a DAS subspace under a->a+1 shift."""
    proj = Q @ Q.T
    a_all = a_vec.cpu()
    b_all = b_vec.cpu()

    ab_to_idx = {}
    for idx in range(len(a_all)):
        ab_to_idx[(a_all[idx].item(), b_all[idx].item())] = idx

    # Project activations into subspace
    projected = activations @ Q  # (n, k)

    deltas = []
    for ti in test_idx:
        ti_val = ti.item()
        a_val = a_all[ti_val].item()
        b_val = b_all[ti_val].item()
        a_shifted = (a_val + 1) % p
        shifted_idx = ab_to_idx.get((a_shifted, b_val))
        if shifted_idx is None:
            continue
        delta = projected[shifted_idx] - projected[ti_val]
        deltas.append(delta)

    if len(deltas) < 10:
        return 0.0

    deltas_tensor = torch.stack(deltas)
    mean_delta = deltas_tensor.mean(dim=0)
    mean_norm = mean_delta.norm().item()
    if mean_norm < 1e-8:
        return 0.0

    deviations = (deltas_tensor - mean_delta).norm(dim=1)
    threshold = 0.1 * mean_norm
    consistent = (deviations < threshold).float().mean().item()
    return consistent


def random_subspace_iia(model, activations, labels, dataset, test_idx, device,
                        a_vec, b_vec, p, k=2, layer=0, n_pairs=200):
    """Random subspace baseline: sample a random Q and measure IIA + equivariance."""
    hook_name = f"blocks.{layer}.hook_resid_post"
    d_model = model.cfg.d_model

    R = torch.randn(d_model, k, device=device)
    Q, _ = torch.linalg.qr(R)
    proj = Q @ Q.T

    test_acts = activations[test_idx]
    test_labels = labels[test_idx]
    test_data = dataset[test_idx]

    n_test = len(test_idx)
    pairs = []
    for i in range(n_test):
        for j in range(i + 1, min(i + 30, n_test)):
            if test_labels[i] != test_labels[j]:
                pairs.append((i, j))
                if len(pairs) >= n_pairs:
                    break
        if len(pairs) >= n_pairs:
            break

    if len(pairs) == 0:
        return 0.0, 0.0

    correct = 0
    total = 0
    with torch.inference_mode():
        for base_i, src_i in tqdm(pairs, desc="random IIA", leave=False):
            ba = test_acts[base_i]
            sa = test_acts[src_i]
            si = test_labels[src_i].item()

            iv = ba - ba @ proj + sa @ proj
            base_tokens = test_data[base_i].unsqueeze(0)

            def hook_fn(act, hook=None, iv_vec=iv):
                new = act.clone()
                new[0, -1, :] = iv_vec
                return new

            logits = model.run_with_hooks(
                base_tokens, fwd_hooks=[(hook_name, hook_fn)]
            )[0, -1, :]
            pred = logits.argmax(dim=-1).item()
            if pred == si:
                correct += 1
            total += 1

    iia = correct / max(total, 1)
    equivariance = eval_das_equivariance(Q, activations, a_vec, b_vec, p, test_idx, device)
    return iia, equivariance


def run_single_operation(operation, device="cuda"):
    """Run the full pipeline for a single operation."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    logger = logging.getLogger(__name__)

    op_config = OPERATIONS[operation]
    p = op_config["p"]
    n_epochs = EPOCH_DEFAULTS.get(operation, 25000)

    logger.info(f"[{utc_ts()}] Starting {operation} (p={p}, epochs={n_epochs})")

    # Step 1: Train grokking model
    logger.info(f"[{utc_ts()}] Training grokking model...")
    model, cfg, test_loss, test_acc, grokked = train_grokking_model(
        operation, p, device, n_epochs=n_epochs,
    )
    logger.info(f"[{utc_ts()}] Test loss={test_loss:.4f}, acc={test_acc:.4f}, grokked={grokked}")

    # Step 2: Cache activations
    logger.info(f"[{utc_ts()}] Caching activations...")
    dataset, labels, train_idx, test_idx, a_vec, b_vec = build_data(operation, p, device)
    activations = cache_all_activations(model, dataset, device, layer=0)

    d_model = model.cfg.d_model
    n_classes = p

    result = {
        "operation": operation,
        "p": p,
        "n_epochs": n_epochs,
        "grokked": grokked,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "timestamp": utc_ts(),
        "vae_results": {},
        "das_results": {},
        "random_control": {},
    }

    # Step 3: For each k value, train VAE + DAS + random baseline
    for k in [2, 4]:
        k_str = f"k{k}"
        logger.info(f"[{utc_ts()}] === k={k} ===")

        # -- VAE --
        logger.info(f"[{utc_ts()}] Training StructuredVAE (z_causal={k}, z_nuisance=15)...")
        vae = build_structured_vae(
            d_input=d_model, z_causal_dim=k, z_nuisance_dim=15,
            hidden_dim=128, n_classes=n_classes,
        )
        vae, vae_loss = train_vae(
            vae, activations, labels, n_classes, device,
            n_epochs=300, batch_size=256, lr=1e-3, alpha=10.0,
        )

        # Classifier accuracy
        cls_acc = eval_vae_classifier(vae, activations[test_idx], labels[test_idx], device)
        logger.info(f"[{utc_ts()}] VAE classifier acc={cls_acc:.4f}")

        # Reconstruction loss
        recon_loss = eval_vae_reconstruction(vae, activations[test_idx], device)
        logger.info(f"[{utc_ts()}] VAE recon loss={recon_loss:.6f}")

        # IIA
        logger.info(f"[{utc_ts()}] Evaluating VAE IIA...")
        vae_iia = eval_encode_swap_decode_iia(
            vae, model, activations, labels, dataset, device,
            test_idx, n_pairs=500, layer=0,
        )
        logger.info(f"[{utc_ts()}] VAE IIA={vae_iia:.4f}")

        # Hard IIA
        logger.info(f"[{utc_ts()}] Evaluating VAE hard IIA...")
        vae_iia_hard = eval_encode_swap_decode_iia_hard(
            vae, model, activations, labels, dataset, device,
            test_idx, n_pairs=200, layer=0,
        )
        logger.info(f"[{utc_ts()}] VAE hard IIA={vae_iia_hard:.4f}")

        # Equivariance
        logger.info(f"[{utc_ts()}] Evaluating VAE equivariance...")
        vae_equiv = eval_equivariance(
            vae, activations, labels, a_vec, b_vec, p, test_idx, device,
        )
        logger.info(f"[{utc_ts()}] VAE equivariance={vae_equiv:.4f}")

        # Linearized subspace
        Q_vae = vae.get_causal_subspace()
        subspace_list = Q_vae.cpu().tolist()

        result["vae_results"][k_str] = {
            "vae_iia": vae_iia,
            "vae_iia_hard": vae_iia_hard,
            "vae_equivariance": vae_equiv,
            "classifier_accuracy": cls_acc,
            "recon_loss": recon_loss,
            "linearized_subspace": subspace_list,
        }

        # -- DAS baseline --
        logger.info(f"[{utc_ts()}] Training DAS baseline k={k}...")
        das_iia, _, Q_das = train_das_baseline(
            model, activations, labels, dataset, test_idx, device,
            k=k, n_steps=400, layer=0, n_pairs=500,
        )
        das_equiv = 0.0
        if Q_das is not None:
            das_equiv = eval_das_equivariance(
                Q_das, activations, a_vec, b_vec, p, test_idx, device,
            )
        logger.info(f"[{utc_ts()}] DAS IIA={das_iia:.4f}, equivariance={das_equiv:.4f}")

        result["das_results"][k_str] = {
            "iia": das_iia,
            "equivariance": das_equiv,
        }

        # -- Random baseline (only for k=2) --
        if k == 2:
            logger.info(f"[{utc_ts()}] Random subspace baseline k={k}...")
            rand_iia, rand_equiv = random_subspace_iia(
                model, activations, labels, dataset, test_idx, device,
                a_vec, b_vec, p, k=k, layer=0, n_pairs=200,
            )
            logger.info(f"[{utc_ts()}] Random IIA={rand_iia:.4f}, equivariance={rand_equiv:.4f}")
            result["random_control"][k_str] = {
                "iia": rand_iia,
                "equivariance": rand_equiv,
            }

    logger.info(f"[{utc_ts()}] {operation} complete.")
    return result


# ===================================================================
# Modal functions
# ===================================================================


@app.function(gpu="A100", timeout=14400, volumes={"/results": results_vol})
def run_operation(operation: str) -> dict:
    """Run the full structured VAE pipeline for one operation on GPU."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    logger = logging.getLogger(__name__)

    t0 = time.time()
    result_dir = f"/results/grassmannian_atlas/vae/{operation}"
    os.makedirs(result_dir, exist_ok=True)

    try:
        result = run_single_operation(operation, device="cuda")
        result["elapsed_seconds"] = round(time.time() - t0, 1)
        result["status"] = "success"
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"[{utc_ts()}] FAILED: {e}\n{tb}")
        result = {
            "operation": operation,
            "status": "error",
            "error": str(e),
            "traceback": tb[-2000:],
            "elapsed_seconds": round(time.time() - t0, 1),
        }

    # Save results to volume
    out_path = os.path.join(result_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"[{utc_ts()}] Saved results to {out_path}")

    results_vol.commit()
    return result


@app.local_entrypoint()
def main(
    operations: str = "",
):
    """Launch structured VAE experiments across operations in parallel.

    Usage:
        modal run --detach experiments/batch6_atlas/06_21_2026_UPDATE/structured_vae_atlas.py

        # Specific operations:
        modal run --detach experiments/batch6_atlas/06_21_2026_UPDATE/structured_vae_atlas.py \
            --operations addition,multiplication
    """
    if operations:
        op_list = [o.strip() for o in operations.split(",")]
    else:
        op_list = list(OPERATIONS.keys())

    print(f"Structured VAE Atlas -- {len(op_list)} operations")
    print(f"  Operations: {op_list}")
    print(f"  Started: {utc_ts()}")
    print()

    t0 = time.time()

    # Spawn all operations in parallel
    handles = []
    for op in op_list:
        h = run_operation.spawn(operation=op)
        handles.append((op, h))
        print(f"  Spawned {op} (A100)")

    print(f"\n{len(handles)} containers spawned. Collecting results...\n")

    # Collect results
    results = []
    for op, h in handles:
        try:
            result = h.get()
        except Exception as e:
            tb = traceback.format_exc()
            result = {
                "operation": op, "status": "error",
                "error": f"{type(e).__name__}: {e}",
                "traceback": tb[-2000:],
            }
        results.append(result)

        status = result.get("status", "unknown")
        elapsed = result.get("elapsed_seconds", 0)
        vae_k2 = result.get("vae_results", {}).get("k2", {})
        das_k2 = result.get("das_results", {}).get("k2", {})

        vae_iia_str = f"{vae_k2.get('vae_iia', 0):.3f}" if vae_k2 else "N/A"
        das_iia_str = f"{das_k2.get('iia', 0):.3f}" if das_k2 else "N/A"

        print(f"  {op:20s}  {status:7s}  {elapsed:6.0f}s  "
              f"VAE_IIA={vae_iia_str}  DAS_IIA={das_iia_str}")
        if status == "error":
            print(f"    ERROR: {result.get('error', 'unknown')[:200]}")

    total = time.time() - t0
    successes = sum(1 for r in results if r.get("status") == "success")
    print(f"\n{'=' * 60}")
    print(f"Structured VAE Atlas complete: {successes}/{len(results)} operations in {total:.0f}s")

    # Summary table
    print(f"\n{'Operation':>20s}  {'Grok':>5s}  {'VAE k2':>7s}  {'VAE k4':>7s}  "
          f"{'DAS k2':>7s}  {'DAS k4':>7s}  {'Rand k2':>7s}  {'Eqv k2':>7s}")
    print("-" * 90)
    for r in results:
        if r.get("status") != "success":
            print(f"{r.get('operation', '?'):>20s}  {'ERR':>5s}")
            continue
        vae2 = r.get("vae_results", {}).get("k2", {})
        vae4 = r.get("vae_results", {}).get("k4", {})
        das2 = r.get("das_results", {}).get("k2", {})
        das4 = r.get("das_results", {}).get("k4", {})
        rnd2 = r.get("random_control", {}).get("k2", {})
        print(f"{r['operation']:>20s}  "
              f"{'Y' if r.get('grokked') else 'N':>5s}  "
              f"{vae2.get('vae_iia', 0):7.3f}  "
              f"{vae4.get('vae_iia', 0):7.3f}  "
              f"{das2.get('iia', 0):7.3f}  "
              f"{das4.get('iia', 0):7.3f}  "
              f"{rnd2.get('iia', 0):7.3f}  "
              f"{vae2.get('vae_equivariance', 0):7.3f}")
