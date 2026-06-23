"""Grokking Nonlinear DSI — Does the grokking causal variable need nonlinear access?

Linear DAS gets 0.0 IIA on the fully-grokked modular addition model. The Nanda
et al. Fourier picture predicts why: the "number identity" variable is encoded as
cos/sin components on a circle — a fundamentally nonlinear representation.

This script trains the grokking model to completion, then applies the nonlinear
DSI ladder (linear → quadratic → MLP featurizer) with random controls at the
FINAL fully-grokked checkpoint. If the MLP featurizer recovers high IIA while
the random control stays flat, that confirms: the causal variable exists but
lives on a curved manifold in activation space, exactly as the Fourier picture
predicts.

Usage:
    python -u experiments/batch6_atlas/grokking_nonlinear_dsi.py \
        --device cuda --n-epochs 25000

    # Load pre-trained checkpoints:
    python -u experiments/batch6_atlas/grokking_nonlinear_dsi.py \
        --device cuda --load-from experiments/results/grokking_checkpoints.pt
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import random
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
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformer_lens import HookedTransformer, HookedTransformerConfig

from factorization_circuits.pipeline.utils.factor_das_kernel import (
    VanillaQ, FactorParam, Site, make_hook, site_resid, eval_iia,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

P = 113
FRAC_TRAIN = 0.3
DATA_SEED = 598


def utc_ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Grokking model + data (reused from grokking_das_emergence.py) ─────

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


def train_grokking_model(device, n_epochs=25000, lr=1e-3, wd=1.0):
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

    for epoch in tqdm(range(n_epochs), desc="training grokking model"):
        train_logits = model(train_data)[:, -1]
        train_loss = F.cross_entropy(train_logits, train_labels)
        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (epoch + 1) % 5000 == 0:
            with torch.no_grad():
                test_logits = model(test_data)[:, -1]
                test_loss = F.cross_entropy(test_logits, test_labels)
            logger.info(f"  Epoch {epoch+1}: train={train_loss.item():.4f} test={test_loss.item():.4f}")

    with torch.no_grad():
        test_logits = model(test_data)[:, -1]
        test_loss = F.cross_entropy(test_logits, test_labels)
        test_acc = (test_logits.argmax(-1) == test_labels).float().mean()
    logger.info(f"  Final: test_loss={test_loss.item():.4f} test_acc={test_acc.item():.3f}")

    return model, cfg


# ── Nonlinear featurizers (same as nonlinear_dsi.py) ──────────────────

class QuadraticQ(FactorParam):
    def __init__(self, d_site: int, k: int, hidden_dim: int = 0, device="cpu"):
        super().__init__()
        self._d = d_site
        self._k = k
        self._hidden = hidden_dim if hidden_dim > 0 else d_site
        self.R = nn.Parameter(torch.randn(d_site, k, device=device) * 0.02)
        self.quad_in = nn.Parameter(torch.randn(d_site, self._hidden, device=device) * 0.02)
        self.quad_out = nn.Parameter(torch.randn(self._hidden, d_site, device=device) * 0.02)

    def forward(self):
        Q, _ = torch.linalg.qr(self.R)
        return Q

    def featurize(self, x):
        quad = (x @ self.quad_in) ** 2
        return x + quad @ self.quad_out

    @property
    def d_site(self): return self._d
    @property
    def k(self): return self._k
    def info(self): return {"type": "quadratic", "d_site": self._d, "k": self._k,
                            "hidden_dim": self._hidden}


class MLPQ(FactorParam):
    def __init__(self, d_site: int, k: int, hidden_dim: int = 128, device="cpu"):
        super().__init__()
        self._d = d_site
        self._k = k
        self._hidden = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(d_site, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_site, bias=True),
        ).to(device)
        self.R = nn.Parameter(torch.randn(d_site, k, device=device) * 0.02)

    def forward(self):
        Q, _ = torch.linalg.qr(self.R)
        return Q

    def featurize(self, x):
        return self.mlp(x)

    @property
    def d_site(self): return self._d
    @property
    def k(self): return self._k
    def info(self): return {"type": "mlp", "d_site": self._d, "k": self._k,
                            "hidden_dim": self._hidden}


# ── Caching + nonlinear training ──────────────────────────────────────

@torch.no_grad()
def cache_grokking_pairs(model, dataset, labels, train_idx, layer, device,
                         n_pairs=200):
    pairs_idx = []
    train_data = dataset[train_idx]
    train_labels = labels[train_idx]
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
        ba = cache_i[hook_name][0, -1, :]
        sa = cache_j[hook_name][0, -1, :]
        si = train_labels[j].item()
        cached.append((tokens_i, ba, sa, si))
    return cached


def make_random_control(cached, rng_seed=99):
    rng = random.Random(rng_seed)
    shuffled = list(cached)
    sources = [sa.clone() for _, _, sa, _ in shuffled]
    rng.shuffle(sources)
    return [(bt, ba, sources[i], si) for i, (bt, ba, _, si) in enumerate(shuffled)]


def train_nonlinear_das(model, site, param, cached, *, n_steps, lr, batch=16,
                        device="cpu"):
    has_featurize = hasattr(param, "featurize")
    opt = torch.optim.Adam(param.parameters(), lr=lr)
    rng = random.Random(0)
    history = []
    frozen = {p for p in model.parameters() if p.requires_grad}
    for p in frozen:
        p.requires_grad_(False)
    try:
        pbar = tqdm(range(n_steps), desc="nonlinear-DAS", leave=False)
        for step in pbar:
            idx = rng.sample(range(len(cached)), min(batch, len(cached)))
            Q = param()
            proj = Q @ Q.T
            loss = torch.tensor(0., device=device)
            for i in idx:
                bt, ba, sa, si = cached[i]
                if has_featurize:
                    fb = param.featurize(ba)
                    fs = param.featurize(sa)
                    iv = ba + (fs @ proj - fb @ proj)
                else:
                    iv = ba - ba @ proj + sa @ proj
                lp = F.log_softmax(
                    model.run_with_hooks(
                        bt, fwd_hooks=[(site.hook_name, make_hook(site, iv))]
                    )[0, -1, :], dim=-1,
                )
                loss = loss - lp[si]
            loss = loss / len(idx)
            loss.backward()
            opt.step()
            opt.zero_grad()
            if step % 50 == 0 or step == n_steps - 1:
                history.append({"step": step, "loss": float(loss.detach().item())})
                pbar.set_postfix(loss=f"{history[-1]['loss']:.3f}")
    finally:
        for p in frozen:
            p.requires_grad_(True)
    with torch.no_grad():
        Q_final = param().detach()
    return Q_final, history


@torch.no_grad()
def eval_nonlinear_iia(model, site, param, Q, cached_eval):
    has_featurize = hasattr(param, "featurize")
    proj = Q @ Q.T
    correct = 0
    for bt, ba, sa, si in cached_eval:
        if has_featurize:
            fb = param.featurize(ba)
            fs = param.featurize(sa)
            iv = ba + (fs @ proj - fb @ proj)
        else:
            iv = ba - ba @ proj + sa @ proj
        lp = F.log_softmax(
            model.run_with_hooks(
                bt, fwd_hooks=[(site.hook_name, make_hook(site, iv))]
            )[0, -1, :], dim=-1,
        )
        blp = F.log_softmax(model(bt)[0, -1, :], dim=-1)
        if lp[si] > blp[si]:
            correct += 1
    return correct / max(len(cached_eval), 1)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Grokking Nonlinear DSI Ladder")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-epochs", type=int, default=25000)
    parser.add_argument("--das-k", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=400)
    parser.add_argument("--load-from", type=str, default=None)
    parser.add_argument("--output-dir", default="experiments/results")
    parser.add_argument("--no-wandb", action="store_true")
    # Compatibility with modal_atlas.py
    parser.add_argument("--model", default="grokking")
    parser.add_argument("--task", default="modular_addition")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--k", type=int, default=None)
    args = parser.parse_args()

    device = args.device
    k = args.k or args.das_k
    n_steps = args.n_steps
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[%s] Grokking Nonlinear DSI — k=%d, n_steps=%d", utc_ts(), k, n_steps)

    # Step 1: Get the fully-grokked model
    if args.load_from and Path(args.load_from).exists():
        logger.info("[%s] Loading checkpoints from %s", utc_ts(), args.load_from)
        saved = torch.load(args.load_from, map_location=device, weights_only=False)
        cfg = saved["config"]
        model = HookedTransformer(cfg)
        model.load_state_dict(saved["checkpoints"][-1])
        model.eval()
    else:
        logger.info("[%s] Training grokking model (%d epochs)...", utc_ts(), args.n_epochs)
        model, cfg = train_grokking_model(device, n_epochs=args.n_epochs)
        model.eval()

    d = model.cfg.d_model  # 128
    dataset, labels, train_idx, test_idx = build_grokking_data(device)

    # Step 2: Cache DAS pairs
    layer = 0
    site = site_resid(layer, d)
    logger.info("[%s] Caching grokking DAS pairs at L%d (d_model=%d)", utc_ts(), layer, d)
    cached = cache_grokking_pairs(model, dataset, labels, train_idx, layer, device, n_pairs=300)
    n_train = int(len(cached) * 0.75)
    cached_train = cached[:n_train]
    cached_eval = cached[n_train:]
    cached_random = make_random_control(cached_train)
    cached_eval_random = make_random_control(cached_eval)

    logger.info("  %d train pairs, %d eval pairs", len(cached_train), len(cached_eval))

    # Step 3: Nonlinearity ladder
    featurizer_configs = [
        ("linear", lambda: VanillaQ(d, k, device=device), False),
        ("quadratic_small", lambda: QuadraticQ(d, k, hidden_dim=k, device=device), True),
        ("quadratic_medium", lambda: QuadraticQ(d, k, hidden_dim=k * 4, device=device), True),
        ("quadratic_full", lambda: QuadraticQ(d, k, hidden_dim=d, device=device), True),
        ("mlp_32", lambda: MLPQ(d, k, hidden_dim=32, device=device), True),
        ("mlp_128", lambda: MLPQ(d, k, hidden_dim=128, device=device), True),
        ("mlp_512", lambda: MLPQ(d, k, hidden_dim=512, device=device), True),
    ]

    results_list = []

    for feat_name, make_param, is_nonlinear in featurizer_configs:
        logger.info("\n[%s] === %s ===", utc_ts(), feat_name)

        # Genuine pairing
        param = make_param()
        n_params = sum(p.numel() for p in param.parameters())
        logger.info("  Parameters: %s", f"{n_params:,}")

        if is_nonlinear:
            Q, hist = train_nonlinear_das(
                model, site, param, cached_train,
                n_steps=n_steps, lr=1e-3, batch=16, device=device,
            )
            iia = eval_nonlinear_iia(model, site, param, Q, cached_eval)
        else:
            from factorization_circuits.pipeline.utils.factor_das_kernel import train_factor_das
            Q, hist = train_factor_das(
                model, site, param, cached_train,
                n_steps=n_steps, lr=1e-3, batch=16, device=device,
            )
            iia = eval_iia(model, site, Q, cached_eval)

        logger.info("  IIA (genuine): %.3f", iia)

        # Random control
        param_ctrl = make_param()
        if is_nonlinear:
            Q_ctrl, hist_ctrl = train_nonlinear_das(
                model, site, param_ctrl, cached_random,
                n_steps=n_steps, lr=1e-3, batch=16, device=device,
            )
            iia_ctrl = eval_nonlinear_iia(model, site, param_ctrl, Q_ctrl, cached_eval_random)
        else:
            Q_ctrl, hist_ctrl = train_factor_das(
                model, site, param_ctrl, cached_random,
                n_steps=n_steps, lr=1e-3, batch=16, device=device,
            )
            iia_ctrl = eval_iia(model, site, Q_ctrl, cached_eval_random)

        logger.info("  IIA (random control): %.3f", iia_ctrl)
        logger.info("  Genuine - Control: %+.3f", iia - iia_ctrl)

        results_list.append({
            "featurizer": feat_name,
            "n_params": n_params,
            "is_nonlinear": is_nonlinear,
            "iia_genuine": iia,
            "iia_random_control": iia_ctrl,
            "iia_gap": iia - iia_ctrl,
            "final_train_loss": hist[-1]["loss"] if hist else None,
            "final_ctrl_loss": hist_ctrl[-1]["loss"] if hist_ctrl else None,
        })

    # Save
    result = {
        "timestamp": utc_ts(),
        "model": "grokking_1layer",
        "task": "modular_addition",
        "p": P,
        "layer": layer,
        "k": k,
        "n_steps": n_steps,
        "d_model": d,
        "ladder": results_list,
    }

    out_file = output_dir / "grokking_nonlinear_dsi.jsonl"
    with open(out_file, "a") as f:
        f.write(json.dumps(result, default=str) + "\n")
    logger.info("[%s] Results appended to %s", utc_ts(), out_file)

    # Summary
    print(f"\n{'='*60}")
    print(f"Grokking Nonlinear DSI Ladder — mod {P}, L{layer}")
    print(f"\n{'Featurizer':<25s} {'Params':>10s} {'IIA':>8s} {'Control':>8s} {'Gap':>8s}")
    print("-" * 65)
    for r in results_list:
        print(f"{r['featurizer']:<25s} {r['n_params']:>10,} {r['iia_genuine']:>8.3f} "
              f"{r['iia_random_control']:>8.3f} {r['iia_gap']:>+8.3f}")

    # Diagnosis
    linear_iia = results_list[0]["iia_genuine"]
    best_nl = max(results_list[1:], key=lambda r: r["iia_genuine"])
    best_nl_ctrl = max(results_list[1:], key=lambda r: r["iia_random_control"])
    print(f"\nLinear DAS IIA: {linear_iia:.3f}")
    print(f"Best nonlinear IIA: {best_nl['iia_genuine']:.3f} ({best_nl['featurizer']})")
    print(f"Best control IIA: {best_nl_ctrl['iia_random_control']:.3f} ({best_nl_ctrl['featurizer']})")

    if best_nl["iia_genuine"] > linear_iia + 0.1 and best_nl_ctrl["iia_random_control"] < linear_iia + 0.05:
        print("\nDIAGNOSIS: NONLINEAR FOURIER VARIABLE — consistent with Nanda et al. circular representation")
    elif best_nl["iia_genuine"] > linear_iia + 0.1 and best_nl_ctrl["iia_random_control"] > linear_iia + 0.05:
        print("\nDIAGNOSIS: SUTTER DILEMMA — capacity overfitting, not genuine nonlinear structure")
    else:
        print("\nDIAGNOSIS: NO CAUSAL VARIABLE FOUND — even nonlinear featurizers fail")


if __name__ == "__main__":
    main()
