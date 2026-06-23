"""Sparse (group-lasso) DAS on standard grokking models.

Applies DAS with group lasso regularization on the residual-stream dimensions
to find sparse axis-aligned directions that encode the causal variable.

For grokked models (addition, multiplication, etc.), surviving dimensions should
correspond to Fourier frequency components (cos/sin pairs).
For non-grokked models (squaring, cubing), sparsity should be unstructured.

Usage:
    modal run --detach experiments/sparse_das_grokking.py
    modal run --detach experiments/sparse_das_grokking.py --operations addition,squaring
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

app = modal.App("sparse-das-grokking", image=image)
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


def cache_das_pairs(model, dataset, labels, test_idx, device, layer=0, n_pairs=500):
    """Cache pairs of (base, source) with different labels for DAS training."""
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
    """Evaluate IIA with intervention directions U (d_model, k)."""
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


def train_sparse_das(
    model, data, layer, device, k=4, n_steps=200, lr=1e-3,
    l1_lambda=0.01, reg_type="group_lasso",
):
    """Train DAS with group lasso on residual-stream dimensions.

    Uses identity as implicit "factor bank" — group lasso on rows of A
    corresponds to selecting which dimensions of the residual stream matter.

    A in R^{d_model x k}. Group lasso penalizes ||A_i||_2 per row i,
    killing entire dimensions. Surviving rows = causally active dimensions.
    """
    d_model = data[0]["base_resid"].shape[0]
    hook_name = f"blocks.{layer}.hook_resid_post"

    deltas = torch.stack([d["source_resid"] - d["base_resid"] for d in data])
    _, _, Vh = torch.linalg.svd(deltas, full_matrices=False)
    A = nn.Parameter(Vh[:k].T.clone().to(device))
    optimizer = torch.optim.Adam([A], lr=lr)

    history = []
    best_ckpt = {"iia": -1.0, "step": 0, "A": None}

    micro_batch = 10
    n_train = min(len(data), 100)
    mb_ranges = list(range(0, n_train, micro_batch))

    for step in tqdm(range(n_steps), desc=f"sparse DAS ({reg_type})", leave=False):
        optimizer.zero_grad()

        for mb_start in mb_ranges:
            Q, _ = torch.linalg.qr(A)
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
                Q_eval, _ = torch.linalg.qr(A)
                iia = eval_iia(model, data, Q_eval, layer, device)
                row_norms = A.norm(dim=1)
                n_active = (row_norms > row_norms.max() * 0.1).sum().item()
                top_dims = torch.argsort(row_norms, descending=True)[:10].tolist()

                entry = {
                    "step": step + 1,
                    "iia": iia,
                    "n_active": n_active,
                    "top_dims": top_dims,
                    "max_row_norm": row_norms.max().item(),
                }
                history.append(entry)

                if iia > best_ckpt["iia"]:
                    best_ckpt = {
                        "iia": iia,
                        "step": step + 1,
                        "A": A.detach().clone(),
                        "n_active": n_active,
                    }

    A_final = best_ckpt["A"] if best_ckpt["A"] is not None else A.detach()
    Q_final, _ = torch.linalg.qr(A_final)

    row_norms = A_final.norm(dim=1)
    surviving_dims = torch.argsort(row_norms, descending=True)
    n_active = (row_norms > row_norms.max() * 0.1).sum().item()

    return Q_final, A_final, surviving_dims.tolist(), n_active, history, best_ckpt


def analyze_fourier_alignment(surviving_dims, model, p, d_model=128):
    """Check if surviving dimensions correspond to Fourier frequencies.

    Grokked models encode numbers as cos(2*pi*f*a/p), sin(2*pi*f*a/p).
    We check if the top surviving dimensions align with these Fourier components
    by computing the embedding matrix's DFT and seeing which frequencies dominate.
    """
    W_E = model.embed.W_E.detach().cpu()  # (p+1, d_model)
    W_E_numbers = W_E[:p]  # exclude equality token

    freqs = torch.fft.fft(W_E_numbers, dim=0)  # (p, d_model) complex
    freq_power = freqs.abs()  # (p, d_model)

    top_dims = surviving_dims[:20]
    fourier_profile = {}
    for dim_idx in top_dims:
        if dim_idx >= d_model:
            continue
        power_at_dim = freq_power[:, dim_idx].numpy()
        dominant_freq = int(np.argmax(power_at_dim[1:p // 2])) + 1
        fourier_profile[int(dim_idx)] = {
            "dominant_freq": dominant_freq,
            "power_at_dominant": float(power_at_dim[dominant_freq]),
            "total_power": float(power_at_dim.sum()),
            "concentration": float(power_at_dim[dominant_freq] / (power_at_dim.sum() + 1e-8)),
        }

    unique_freqs = set(v["dominant_freq"] for v in fourier_profile.values())
    mean_concentration = np.mean([v["concentration"] for v in fourier_profile.values()]) if fourier_profile else 0.0

    return {
        "top_dim_fourier_profile": fourier_profile,
        "n_unique_frequencies": len(unique_freqs),
        "unique_frequencies": sorted(unique_freqs),
        "mean_fourier_concentration": float(mean_concentration),
    }


def plot_sparsity_profile(A_final, operation, grokked, result_dir):
    """Plot row norms of A showing which dimensions survive."""
    row_norms = A_final.norm(dim=1).cpu().numpy()
    sorted_norms = np.sort(row_norms)[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(range(len(sorted_norms)), sorted_norms, color="steelblue", alpha=0.7)
    axes[0].set_xlabel("Dimension (sorted by importance)")
    axes[0].set_ylabel("Row norm ||A_i||")
    axes[0].set_title(f"{operation} ({'grokked' if grokked else 'NOT grokked'})")

    threshold = row_norms.max() * 0.1
    n_active = (row_norms > threshold).sum()
    axes[0].axhline(y=threshold, color="red", linestyle="--", alpha=0.5, label=f"10% threshold ({n_active} active)")
    axes[0].legend()

    axes[1].bar(range(len(row_norms)), row_norms, color="steelblue", alpha=0.7)
    axes[1].set_xlabel("Residual stream dimension index")
    axes[1].set_ylabel("Row norm ||A_i||")
    axes[1].set_title(f"Dimension importance (unsorted)")

    plt.tight_layout()
    fig.savefig(os.path.join(result_dir, f"sparsity_profile_{operation}.png"), dpi=150)
    plt.close(fig)


def _save_incremental(result, save_path, vol=None):
    """Write partial results to disk (and commit volume if provided)."""
    if save_path is None:
        return
    with open(save_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    if vol is not None:
        vol.commit()
    logger.info(f"[{utc_ts()}] Incremental save -> {save_path}")


def run_single_operation(operation, device="cuda", save_path=None, vol=None):
    """Run sparse DAS for a single operation."""
    p = OPERATIONS[operation]["p"]
    n_epochs = EPOCH_DEFAULTS.get(operation, 25000)

    logger.info(f"[{utc_ts()}] Starting {operation} (p={p}, epochs={n_epochs})")

    model, cfg, test_loss, test_acc, grokked = train_grokking_model(
        operation, p, device, n_epochs=n_epochs,
    )
    logger.info(f"[{utc_ts()}] Model trained: test_loss={test_loss:.4f}, acc={test_acc:.4f}, grokked={grokked}")

    dataset, labels, train_idx, test_idx, a_vec, b_vec = build_data(operation, p, device)

    logger.info(f"[{utc_ts()}] Caching DAS pairs...")
    pairs = cache_das_pairs(model, dataset, labels, test_idx, device, n_pairs=500)
    logger.info(f"[{utc_ts()}] Cached {len(pairs)} pairs")

    result = {
        "operation": operation,
        "p": p,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "grokked": grokked,
        "n_pairs": len(pairs),
        "sparse_das_results": {},
    }
    _save_incremental(result, save_path, vol)

    for k in [2, 4, 8]:
        for l1_lam in [0.001, 0.01, 0.05, 0.1]:
            run_key = f"k{k}_lam{l1_lam}"
            logger.info(f"[{utc_ts()}] Sparse DAS k={k}, lambda={l1_lam}...")

            Q, A_final, surviving_dims, n_active, history, best_ckpt = train_sparse_das(
                model, pairs, layer=0, device=device,
                k=k, n_steps=200, lr=1e-3,
                l1_lambda=l1_lam, reg_type="group_lasso",
            )

            iia_final = eval_iia(model, pairs, Q, 0, device)
            fourier_info = analyze_fourier_alignment(surviving_dims, model, p)

            logger.info(
                f"[{utc_ts()}]   IIA={iia_final:.4f}, n_active={n_active}, "
                f"unique_freqs={fourier_info['n_unique_frequencies']}, "
                f"concentration={fourier_info['mean_fourier_concentration']:.4f}"
            )

            result["sparse_das_results"][run_key] = {
                "k": k,
                "l1_lambda": l1_lam,
                "iia": iia_final,
                "best_iia": best_ckpt["iia"],
                "n_active_dims": n_active,
                "surviving_dims_top20": surviving_dims[:20],
                "fourier_analysis": fourier_info,
                "history": history,
            }
            _save_incremental(result, save_path, vol)

    # Also run vanilla DAS (no sparsity) as baseline
    for k in [2, 4]:
        logger.info(f"[{utc_ts()}] Vanilla DAS k={k} (baseline)...")
        Q_vanilla, A_vanilla, _, _, _, best_vanilla = train_sparse_das(
            model, pairs, layer=0, device=device,
            k=k, n_steps=200, lr=1e-3,
            l1_lambda=0.0, reg_type="l1",
        )
        iia_vanilla = eval_iia(model, pairs, Q_vanilla, 0, device)
        result["sparse_das_results"][f"vanilla_k{k}"] = {
            "k": k, "l1_lambda": 0.0,
            "iia": iia_vanilla,
            "best_iia": best_vanilla["iia"],
            "n_active_dims": 128,
        }
        _save_incremental(result, save_path, vol)

    return result, model


# ===================================================================
# Modal functions
# ===================================================================

@app.function(gpu="A100", timeout=14400, volumes={"/results": results_vol})
def run_operation(operation: str) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    t0 = time.time()
    result_dir = f"/results/grassmannian_atlas/sparse_das/{operation}"
    os.makedirs(result_dir, exist_ok=True)

    out_path = os.path.join(result_dir, "results.json")

    try:
        result, model = run_single_operation(
            operation, device="cuda", save_path=out_path, vol=results_vol,
        )
        result["elapsed_seconds"] = round(time.time() - t0, 1)
        result["status"] = "success"
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
def main(operations: str = ""):
    if operations:
        op_list = [o.strip() for o in operations.split(",")]
    else:
        op_list = list(OPERATIONS.keys())

    print(f"Sparse DAS Grokking -- {len(op_list)} operations")
    print(f"  Operations: {op_list}")
    print(f"  Started: {utc_ts()}")
    print()

    t0 = time.time()
    handles = []
    for op in op_list:
        h = run_operation.spawn(operation=op)
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
        best_k2 = result.get("sparse_das_results", {}).get("k2_lam0.01", {})
        print(f"  {op:20s}  {status:7s}  grok={grok}  {elapsed:6.0f}s  "
              f"IIA={best_k2.get('iia', 0):.3f}  active={best_k2.get('n_active_dims', '?')}")

    total = time.time() - t0
    successes = sum(1 for r in results if r.get("status") == "success")
    print(f"\nSparse DAS complete: {successes}/{len(results)} in {total:.0f}s")
