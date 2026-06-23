"""Sparse Structured VAE — JumpReLU / L1 / TopK sparsity on z_causal

Tests whether adding sparsity to the causal latent improves interpretability
(which dimensions activate?) while maintaining IIA performance.

Runs on 4 representative operations spanning all three Grassmannian classes:
  - addition (always Grassmannian)
  - multiplication (always Grassmannian, harder)
  - squaring (never Grassmannian)
  - power (stochastic)

For each operation, trains:
  1. Dense structured VAE (baseline)
  2. JumpReLU-sparse VAE (learned per-dim threshold, L0 penalty)
  3. L1-sparse VAE (L1 penalty on z_causal activations)
  4. TopK-sparse VAE (hard top-4 mask with STE)

Key measurements beyond IIA:
  - L0: average number of active z_causal dimensions per input
  - Factor stability: do the SAME dimensions activate for inputs with the same label?
  - Grassmannian class prediction: always-Grassmannian should need fewer active dims

Usage:
    modal run --detach experiments/batch6_atlas/06_21_2026_UPDATE/sparse_structured_vae.py
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

app = modal.App("sparse-structured-vae", image=image)
results_vol = modal.Volume.from_name("fc-results", create_if_missing=True)

P = 113
FRAC_TRAIN = 0.3
DATA_SEED = 598

OPERATIONS = {
    "addition": {"p": 113, "epochs": 25000, "class": "always"},
    "multiplication": {"p": 113, "epochs": 40000, "class": "always"},
    "squaring": {"p": 113, "epochs": 60000, "class": "never"},
    "power": {"p": 113, "epochs": 80000, "class": "stochastic"},
}

SPARSITY_TYPES = ["dense", "jumprelu", "l1", "topk"]


def utc_ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ===================================================================
# Data and grokking model (same as structured_vae_atlas.py)
# ===================================================================


def compute_labels(a_vec, b_vec, operation, p):
    if operation == "addition":
        return (a_vec + b_vec) % p
    elif operation == "multiplication":
        return (a_vec * b_vec) % p
    elif operation == "squaring":
        return (a_vec * a_vec) % p
    elif operation == "power":
        return torch.tensor(
            [pow(int(a.item()), int(b.item()), p) for a, b in zip(a_vec, b_vec)]
        ).long()
    else:
        raise ValueError(f"Unknown operation: {operation}")


def build_data(operation, p, device):
    is_unary = operation in ("squaring",)
    excludes_zero = operation in ("multiplication", "power")

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


def cache_all_activations(model, dataset, device, layer=0):
    hook_name = f"blocks.{layer}.hook_resid_post"
    all_acts = []
    batch_size = 256
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        with torch.inference_mode():
            _, cache = model.run_with_cache(batch, names_filter=[hook_name])
        all_acts.append(cache[hook_name][:, -1, :].clone())
    return torch.cat(all_acts, dim=0)


# ===================================================================
# Sparse Structured VAE variants
# ===================================================================


def build_sparse_vae(d_input, z_causal_dim, z_nuisance_dim, hidden_dim, n_classes,
                     sparsity_type="dense", topk_k=4, jump_init=0.5):
    class SparseStructuredVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.z_causal_dim = z_causal_dim
            self.z_nuisance_dim = z_nuisance_dim
            self.sparsity_type = sparsity_type
            self.topk_k = topk_k

            z_dim = z_causal_dim + z_nuisance_dim

            self.enc_trunk = nn.Sequential(
                nn.Linear(d_input, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            )
            self.enc_causal_mu = nn.Linear(hidden_dim, z_causal_dim)
            self.enc_causal_logvar = nn.Linear(hidden_dim, z_causal_dim)
            self.enc_nuisance_mu = nn.Linear(hidden_dim, z_nuisance_dim)
            self.enc_nuisance_logvar = nn.Linear(hidden_dim, z_nuisance_dim)

            self.decoder = nn.Sequential(
                nn.Linear(z_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, d_input),
            )
            self.classifier = nn.Linear(z_causal_dim, n_classes)

            if sparsity_type == "jumprelu":
                self.threshold = nn.Parameter(torch.full((z_causal_dim,), jump_init))

        def sparsify(self, z_c):
            if self.sparsity_type == "dense":
                return z_c
            elif self.sparsity_type == "jumprelu":
                mask = (z_c > self.threshold).float()
                return z_c * mask
            elif self.sparsity_type == "l1":
                return z_c
            elif self.sparsity_type == "topk":
                _, top_idx = z_c.abs().topk(self.topk_k, dim=-1)
                mask = torch.zeros_like(z_c)
                mask.scatter_(-1, top_idx, 1.0)
                return z_c * mask
            return z_c

        def encode(self, x):
            h = self.enc_trunk(x)
            return (self.enc_causal_mu(h), self.enc_causal_logvar(h),
                    self.enc_nuisance_mu(h), self.enc_nuisance_logvar(h))

        def reparameterize(self, mu, logvar):
            return mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)

        def forward(self, x):
            mu_c, lv_c, mu_n, lv_n = self.encode(x)
            z_c = self.reparameterize(mu_c, lv_c)
            z_c_sparse = self.sparsify(z_c)
            z_n = self.reparameterize(mu_n, lv_n)
            z = torch.cat([z_c_sparse, z_n], dim=-1)
            x_recon = self.decoder(z)
            logits = self.classifier(z_c_sparse)
            return x_recon, logits, mu_c, lv_c, mu_n, lv_n, z_c_sparse

        def encode_sparse(self, x):
            mu_c, _, mu_n, _ = self.encode(x)
            return self.sparsify(mu_c), mu_n

        def get_causal_subspace(self):
            W1 = self.enc_trunk[0].weight.detach()
            W2 = self.enc_trunk[2].weight.detach()
            W_mu = self.enc_causal_mu.weight.detach()
            W_composed = W_mu @ W2 @ W1
            Q, _ = torch.linalg.qr(W_composed.T)
            return Q[:, :z_causal_dim]

    return SparseStructuredVAE()


def train_sparse_vae(vae, activations, labels, device,
                     n_epochs=300, batch_size=256, lr=1e-3,
                     alpha=10.0, lambda_sparse=1.0, beta=1.0):
    vae = vae.to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    n = len(activations)

    for epoch in tqdm(range(n_epochs), desc=f"VAE ({vae.sparsity_type})", leave=False):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            x, y = activations[idx], labels[idx]

            x_recon, logits, mu_c, lv_c, mu_n, lv_n, z_c_sparse = vae(x)

            recon = F.mse_loss(x_recon, x)
            kl_c = -0.5 * (1 + lv_c - mu_c.pow(2) - lv_c.exp()).mean()
            kl_n = -0.5 * (1 + lv_n - mu_n.pow(2) - lv_n.exp()).mean()
            ce = F.cross_entropy(logits, y)
            loss = recon + beta * (kl_c + kl_n) + alpha * ce

            if vae.sparsity_type == "l1":
                loss = loss + lambda_sparse * z_c_sparse.abs().mean()
            elif vae.sparsity_type == "jumprelu":
                l0 = (z_c_sparse != 0).float().sum(-1).mean()
                loss = loss + lambda_sparse * l0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    vae.eval()
    return vae


def eval_iia(vae, model, activations, labels, dataset, test_idx, device,
             n_pairs=500, layer=0):
    vae.eval()
    hook_name = f"blocks.{layer}.hook_resid_post"
    test_acts = activations[test_idx]
    test_labels = labels[test_idx]
    test_data = dataset[test_idx]

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

    if not pairs:
        return 0.0

    correct = 0
    with torch.inference_mode():
        for base_i, src_i in tqdm(pairs, desc="IIA", leave=False):
            base_act = test_acts[base_i].unsqueeze(0)
            src_act = test_acts[src_i].unsqueeze(0)
            src_label = test_labels[src_i].item()

            z_c_base, z_n_base = vae.encode_sparse(base_act)
            z_c_src, _ = vae.encode_sparse(src_act)

            z_iv = torch.cat([z_c_src, z_n_base], dim=-1)
            h_iv = vae.decoder(z_iv).squeeze(0)

            base_tokens = test_data[base_i].unsqueeze(0)

            def hook_fn(act, hook=None, iv=h_iv):
                new = act.clone()
                new[0, -1, :] = iv
                return new

            logits = model.run_with_hooks(
                base_tokens, fwd_hooks=[(hook_name, hook_fn)]
            )[0, -1, :]
            if logits.argmax().item() == src_label:
                correct += 1

    return correct / len(pairs)


def eval_hard_iia(vae, model, activations, labels, dataset, test_idx, device,
                  n_pairs=200, layer=0):
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
            if min(abs(li - lj), p - abs(li - lj)) > p // 4:
                pairs.append((i, j))
                if len(pairs) >= n_pairs:
                    break
        if len(pairs) >= n_pairs:
            break

    if not pairs:
        return 0.0

    correct = 0
    with torch.inference_mode():
        for base_i, src_i in tqdm(pairs, desc="hard IIA", leave=False):
            base_act = test_acts[base_i].unsqueeze(0)
            src_act = test_acts[src_i].unsqueeze(0)
            src_label = test_labels[src_i].item()

            z_c_base, z_n_base = vae.encode_sparse(base_act)
            z_c_src, _ = vae.encode_sparse(src_act)

            z_iv = torch.cat([z_c_src, z_n_base], dim=-1)
            h_iv = vae.decoder(z_iv).squeeze(0)

            base_tokens = test_data[base_i].unsqueeze(0)

            def hook_fn(act, hook=None, iv=h_iv):
                new = act.clone()
                new[0, -1, :] = iv
                return new

            logits = model.run_with_hooks(
                base_tokens, fwd_hooks=[(hook_name, hook_fn)]
            )[0, -1, :]
            if logits.argmax().item() == src_label:
                correct += 1

    return correct / len(pairs)


def measure_sparsity_stats(vae, activations, labels, test_idx, device):
    """Measure L0, factor stability, and per-dim activation frequency."""
    vae.eval()
    test_acts = activations[test_idx]
    test_labels = labels[test_idx]

    with torch.inference_mode():
        z_c_sparse, _ = vae.encode_sparse(test_acts)

    active = (z_c_sparse.abs() > 1e-8).float()
    l0 = active.sum(-1).mean().item()

    dim_freq = active.mean(dim=0).cpu().tolist()

    unique_labels = test_labels.unique()
    stabilities = []
    for lab in unique_labels:
        mask = test_labels == lab
        if mask.sum() < 3:
            continue
        patterns = active[mask]
        mean_pattern = patterns.mean(dim=0)
        stability = (mean_pattern * (1 - mean_pattern)).mean().item()
        stabilities.append(1.0 - stability)

    factor_stability = sum(stabilities) / max(len(stabilities), 1)

    active_dims = [i for i, f in enumerate(dim_freq) if f > 0.05]
    dead_dims = [i for i, f in enumerate(dim_freq) if f < 0.01]

    return {
        "l0": l0,
        "factor_stability": factor_stability,
        "dim_frequencies": dim_freq,
        "n_active_dims": len(active_dims),
        "n_dead_dims": len(dead_dims),
        "active_dims": active_dims,
    }


def eval_equivariance(vae, activations, a_vec, b_vec, p, test_idx, device):
    vae.eval()
    a_all = a_vec.cpu()
    b_all = b_vec.cpu()
    ab_to_idx = {}
    for idx in range(len(a_all)):
        ab_to_idx[(a_all[idx].item(), b_all[idx].item())] = idx

    with torch.inference_mode():
        z_c_sparse, _ = vae.encode_sparse(activations)

    deltas = []
    for ti in test_idx:
        ti_val = ti.item()
        a_val, b_val = a_all[ti_val].item(), b_all[ti_val].item()
        shifted_idx = ab_to_idx.get(((a_val + 1) % p, b_val))
        if shifted_idx is None:
            continue
        deltas.append(z_c_sparse[shifted_idx] - z_c_sparse[ti_val])

    if len(deltas) < 10:
        return 0.0

    deltas_t = torch.stack(deltas)
    mean_delta = deltas_t.mean(dim=0)
    mean_norm = mean_delta.norm().item()
    if mean_norm < 1e-8:
        return 0.0

    deviations = (deltas_t - mean_delta).norm(dim=1)
    return (deviations < 0.1 * mean_norm).float().mean().item()


# ===================================================================
# Modal function
# ===================================================================


@app.function(gpu="A100", timeout=28800, volumes={"/results": results_vol})
def run_sparse_vae_experiment() -> dict:
    """Run all 4 operations × 4 sparsity types on one A100."""
    import torch

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger(__name__)

    DEVICE = "cuda"
    Z_CAUSAL = 16
    Z_NUISANCE = 16
    HIDDEN = 128
    TOPK_K = 4
    t0 = time.time()

    all_results = {"timestamp": utc_ts(), "z_causal_dim": Z_CAUSAL}

    for op_name, op_cfg in OPERATIONS.items():
        p = op_cfg["p"]
        n_epochs = op_cfg["epochs"]
        grass_class = op_cfg["class"]

        log.info(f"[{utc_ts()}] {'=' * 50}")
        log.info(f"[{utc_ts()}] Operation: {op_name} (class={grass_class}, epochs={n_epochs})")

        # Train grokking model
        log.info(f"[{utc_ts()}] Training grokking model...")
        model, cfg, test_loss, test_acc, grokked = train_grokking_model(
            op_name, p, DEVICE, n_epochs=n_epochs,
        )
        log.info(f"[{utc_ts()}] test_acc={test_acc:.4f}, grokked={grokked}")

        # Cache activations
        dataset, labels, train_idx, test_idx, a_vec, b_vec = build_data(op_name, p, DEVICE)
        activations = cache_all_activations(model, dataset, DEVICE)
        n_classes = p

        op_results = {
            "grokked": grokked,
            "test_accuracy": test_acc,
            "grassmannian_class": grass_class,
            "variants": {},
        }

        for sp_type in SPARSITY_TYPES:
            log.info(f"[{utc_ts()}] --- {op_name} / {sp_type} ---")

            vae = build_sparse_vae(
                d_input=128, z_causal_dim=Z_CAUSAL, z_nuisance_dim=Z_NUISANCE,
                hidden_dim=HIDDEN, n_classes=n_classes,
                sparsity_type=sp_type, topk_k=TOPK_K,
            )
            vae = train_sparse_vae(
                vae, activations, labels, DEVICE,
                n_epochs=300, alpha=10.0,
                lambda_sparse=0.5 if sp_type in ("l1", "jumprelu") else 0.0,
            )

            # Classifier accuracy
            vae.eval()
            with torch.inference_mode():
                z_c_s, _ = vae.encode_sparse(activations[test_idx])
                cls_logits = vae.classifier(z_c_s)
                cls_acc = (cls_logits.argmax(-1) == labels[test_idx]).float().mean().item()
            log.info(f"[{utc_ts()}]   cls_acc={cls_acc:.4f}")

            # Reconstruction
            with torch.inference_mode():
                x_r, _, _, _, _, _, _ = vae(activations[test_idx])
                recon = F.mse_loss(x_r, activations[test_idx]).item()
            log.info(f"[{utc_ts()}]   recon={recon:.6f}")

            # IIA
            iia = eval_iia(vae, model, activations, labels, dataset, test_idx, DEVICE)
            log.info(f"[{utc_ts()}]   IIA={iia:.4f}")

            hard_iia = eval_hard_iia(vae, model, activations, labels, dataset, test_idx, DEVICE)
            log.info(f"[{utc_ts()}]   hard_IIA={hard_iia:.4f}")

            # Equivariance
            equiv = eval_equivariance(vae, activations, a_vec, b_vec, p, test_idx, DEVICE)
            log.info(f"[{utc_ts()}]   equivariance={equiv:.4f}")

            # Sparsity stats
            sparse_stats = measure_sparsity_stats(vae, activations, labels, test_idx, DEVICE)
            log.info(f"[{utc_ts()}]   L0={sparse_stats['l0']:.2f}, "
                     f"stability={sparse_stats['factor_stability']:.4f}, "
                     f"active_dims={sparse_stats['n_active_dims']}/{Z_CAUSAL}")

            op_results["variants"][sp_type] = {
                "classifier_accuracy": cls_acc,
                "reconstruction_mse": recon,
                "iia": iia,
                "hard_iia": hard_iia,
                "equivariance": equiv,
                **sparse_stats,
            }

            # Clean up
            del vae
            torch.cuda.empty_cache()

        all_results[op_name] = op_results

        # Clean up model
        del model, activations
        torch.cuda.empty_cache()

    all_results["elapsed_seconds"] = round(time.time() - t0, 1)

    # Save
    save_dir = "/results/grassmannian_atlas/sparse_vae"
    os.makedirs(save_dir, exist_ok=True)
    out_path = f"{save_dir}/results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    results_vol.commit()
    log.info(f"[{utc_ts()}] Saved to {out_path}")

    # Summary table
    log.info(f"\n{'=' * 100}")
    log.info(f"{'Op':>15s}  {'Class':>10s}  {'Type':>8s}  {'IIA':>6s}  {'Hard':>6s}  "
             f"{'L0':>5s}  {'Stab':>6s}  {'Cls':>5s}  {'Active':>6s}")
    log.info("-" * 85)
    for op_name, op_r in all_results.items():
        if not isinstance(op_r, dict) or "variants" not in op_r:
            continue
        for sp_type, v in op_r["variants"].items():
            log.info(f"{op_name:>15s}  {op_r['grassmannian_class']:>10s}  {sp_type:>8s}  "
                     f"{v['iia']:6.3f}  {v['hard_iia']:6.3f}  "
                     f"{v['l0']:5.1f}  {v['factor_stability']:6.3f}  "
                     f"{v['classifier_accuracy']:5.3f}  "
                     f"{v['n_active_dims']:3d}/{Z_CAUSAL}")

    log.info(f"\nTotal elapsed: {all_results['elapsed_seconds']:.0f}s")
    return all_results


@app.local_entrypoint()
def main():
    handle = run_sparse_vae_experiment.spawn()
    print(f"[{utc_ts()}] Spawned sparse structured VAE experiment")
    print(f"  Handle: {handle.object_id}")
    print(f"  4 ops × 4 sparsity types = 16 VAE trainings")
    print(f"  Results: /results/grassmannian_atlas/sparse_vae/results.json")
