"""Nonlinear DSI Ladder — Controlled nonlinearity sweep for causal variable recovery.

Tests whether linear DAS failures (GT at L8, grokking) are linearity failures
by sweeping featurizer capacity:

  1. Linear DAS (k-dim rotation Q, standard)
  2. Low-rank quadratic (x -> Q @ [x; x^2][:k], quadratic featurizer)
  3. Small MLP featurizer (x -> Q @ MLP(x), 1 hidden layer)

For each, trains to maximize IIA via interchange intervention and reports IIA
as a function of featurizer capacity. A random-init control (same architecture,
random data pairing) distinguishes genuine structure from expressive overfitting.

This directly probes the Sutter et al. dilemma: if IIA climbs smoothly with
capacity AND the random control also climbs, the intervention is vacuous. If
IIA jumps at low capacity while control stays flat, there's a constrained
nonlinear variable.

Usage:
    python -u experiments/batch6_atlas/nonlinear_dsi.py \
        --model gpt2 --task greater_than --layer 8 --k 16 --device cuda

    python -u experiments/batch6_atlas/nonlinear_dsi.py \
        --model gpt2 --task ioi --layer 10 --k 16 --device cuda
"""
from __future__ import annotations

import argparse
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformer_lens import HookedTransformer

from factorization_circuits.pipeline.utils.factor_das_kernel import (
    VanillaQ, Site, cache_pairs, eval_iia, make_hook, site_resid,
    train_factor_das, FactorParam,
)
from factorization_circuits.tasks.prompts import build_task

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

TASK_LAYERS = {
    "sva": 8, "greater_than": 8, "ioi": 10, "gendered_pronoun": 9,
    "capital_country": 8, "gender_bias": 8, "hypernymy": 8,
}


def utc_ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Nonlinear featurizers ──────────────────────────────────────────────

class QuadraticQ(FactorParam):
    """Low-rank quadratic featurizer: f(x) = x + W_out @ (W_in @ x)^2.

    Adds quadratic features back into the original d_site space via a
    learned projection. Output is always (d_site,) so it's compatible
    with the (d_site, d_site) projection matrix Q @ Q.T.
    """
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
        """x: (d_site,) -> features: (d_site,)"""
        quad = (x @ self.quad_in) ** 2
        return x + quad @ self.quad_out

    @property
    def d_site(self): return self._d
    @property
    def k(self): return self._k
    def info(self): return {"type": "quadratic", "d_site": self._d, "k": self._k,
                            "hidden_dim": self._hidden}


class MLPQ(FactorParam):
    """MLP featurizer: h = MLP(x), then Q = orth(W) in h-space.

    The MLP maps d_site -> hidden_dim -> d_site, then a standard rotation
    Q selects k directions. The nonlinearity is in the featurizer, not Q.
    """
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
        """x: (d_site,) -> nonlinear features: (d_site,)"""
        return self.mlp(x)

    @property
    def d_site(self): return self._d
    @property
    def k(self): return self._k
    def info(self): return {"type": "mlp", "d_site": self._d, "k": self._k,
                            "hidden_dim": self._hidden}


# ── Nonlinear training loop ────────────────────────────────────────────

def train_nonlinear_das(model, site, param, cached, *, n_steps, lr, batch=8,
                        device="cpu"):
    """Like train_factor_das but uses param.featurize() for nonlinear
    intervention: iv = f(base) - proj(f(base)) + proj(f(source)) where
    f is the learned featurizer and proj = Q @ Q.T."""
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
    """Eval IIA using the nonlinear featurizer."""
    has_featurize = hasattr(param, "featurize")
    proj = Q @ Q.T
    correct = 0
    for bt, ba, sa, si in tqdm(cached_eval, desc="eval-IIA", leave=False):
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


def make_random_control(cached, rng_seed=99):
    """Shuffle source activations to break causal pairing — same distribution,
    no structure. Used to check if IIA gains are from expressive power alone."""
    rng = random.Random(rng_seed)
    shuffled = list(cached)
    sources = [sa.clone() for _, _, sa, _ in shuffled]
    rng.shuffle(sources)
    return [(bt, ba, sources[i], si) for i, (bt, ba, _, si) in enumerate(shuffled)]


def main():
    parser = argparse.ArgumentParser(description="Nonlinear DSI Ladder")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--task", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--n-steps", type=int, default=400)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="experiments/results")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    device = args.device
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[{utc_ts()}] Loading {args.model}...")
    model = HookedTransformer.from_pretrained(args.model, device=device)
    model.eval()

    logger.info(f"[{utc_ts()}] Building task: {args.task}")
    tokenizer = model.tokenizer
    prompts = build_task(args.task, tokenizer=tokenizer, n_prompts=800)
    rng = random.Random(42)
    rng.shuffle(prompts)
    pairs = []
    for i in range(0, len(prompts) - 1, 2):
        if prompts[i].target_correct != prompts[i + 1].target_correct:
            pairs.append((prompts[i], prompts[i + 1]))
        if len(pairs) >= 200:
            break

    site = site_resid(args.layer, model.cfg.d_model)
    n_train, n_eval = 150, min(50, len(pairs) - 150)
    train_pairs = pairs[:n_train]
    eval_pairs = pairs[n_train:n_train + n_eval]

    logger.info(f"[{utc_ts()}] Caching {n_train} train + {n_eval} eval pairs at L{args.layer}")
    cached_train = cache_pairs(model, tokenizer, train_pairs, site, device)
    cached_eval = cache_pairs(model, tokenizer, eval_pairs, site, device)
    cached_random = make_random_control(cached_train)
    cached_eval_random = make_random_control(cached_eval)

    d = model.cfg.d_model
    k = args.k

    # Define the nonlinearity ladder
    featurizer_configs = [
        ("linear", lambda: VanillaQ(d, k, device=device), False),
        ("quadratic_small", lambda: QuadraticQ(d, k, hidden_dim=k, device=device), True),
        ("quadratic_medium", lambda: QuadraticQ(d, k, hidden_dim=k * 4, device=device), True),
        ("quadratic_full", lambda: QuadraticQ(d, k, hidden_dim=d, device=device), True),
        ("mlp_64", lambda: MLPQ(d, k, hidden_dim=64, device=device), True),
        ("mlp_256", lambda: MLPQ(d, k, hidden_dim=256, device=device), True),
        ("mlp_768", lambda: MLPQ(d, k, hidden_dim=d, device=device), True),
    ]

    results_list = []

    for feat_name, make_param, is_nonlinear in featurizer_configs:
        logger.info(f"\n[{utc_ts()}] === {feat_name} ===")

        # Genuine pairing
        param = make_param()
        n_params = sum(p.numel() for p in param.parameters())
        logger.info(f"  Parameters: {n_params:,}")

        if is_nonlinear:
            Q, hist = train_nonlinear_das(
                model, site, param, cached_train,
                n_steps=args.n_steps, lr=1e-3, batch=8, device=device,
            )
            iia = eval_nonlinear_iia(model, site, param, Q, cached_eval)
        else:
            Q, hist = train_factor_das(
                model, site, param, cached_train,
                n_steps=args.n_steps, lr=1e-3, batch=8, device=device,
            )
            iia = eval_iia(model, site, Q, cached_eval)

        logger.info(f"  IIA (genuine): {iia:.3f}")

        # Random control — same architecture, shuffled pairings
        param_ctrl = make_param()
        if is_nonlinear:
            Q_ctrl, hist_ctrl = train_nonlinear_das(
                model, site, param_ctrl, cached_random,
                n_steps=args.n_steps, lr=1e-3, batch=8, device=device,
            )
            iia_ctrl = eval_nonlinear_iia(
                model, site, param_ctrl, Q_ctrl, cached_eval_random,
            )
        else:
            Q_ctrl, hist_ctrl = train_factor_das(
                model, site, param_ctrl, cached_random,
                n_steps=args.n_steps, lr=1e-3, batch=8, device=device,
            )
            iia_ctrl = eval_iia(model, site, Q_ctrl, cached_eval_random)

        logger.info(f"  IIA (random control): {iia_ctrl:.3f}")
        logger.info(f"  Genuine - Control: {iia - iia_ctrl:+.3f}")

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
        "model": args.model,
        "task": args.task,
        "layer": args.layer,
        "k": k,
        "n_steps": args.n_steps,
        "d_model": d,
        "ladder": results_list,
    }

    out_file = output_dir / "nonlinear_dsi.jsonl"
    with open(out_file, "a") as f:
        f.write(json.dumps(result, default=str) + "\n")
    logger.info(f"[{utc_ts()}] Results appended to {out_file}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Nonlinear DSI Ladder — {args.model} / {args.task} L{args.layer}")
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
        print("\nDIAGNOSIS: CONSTRAINED NONLINEAR VARIABLE — genuine structure, not expressive overfitting")
    elif best_nl["iia_genuine"] > linear_iia + 0.1 and best_nl_ctrl["iia_random_control"] > linear_iia + 0.05:
        print("\nDIAGNOSIS: SUTTER DILEMMA — IIA climbs with capacity for both genuine and random")
    else:
        print("\nDIAGNOSIS: LINEAR IS SUFFICIENT — nonlinearity doesn't help")


if __name__ == "__main__":
    main()
