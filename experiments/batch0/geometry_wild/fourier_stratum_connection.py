"""Fourier-Stratum Connection — do Fourier modes explain stratum classification?

Neel Nanda's grokking work shows that grokked models learn discrete Fourier
features in their embeddings and neuron weights. This experiment connects
that observation to the stratum framework:

  - M_1 (single direction) should correspond to a single Fourier frequency
  - M_k (subspace) should correspond to k/2 Fourier frequencies (cos+sin pairs)
  - M_D (distributed) should have no clean Fourier structure

For each grokked model, compute:
  1. DFT of the learned embedding matrix -> which frequencies are active
  2. DFT of each neuron's weight pattern -> frequency selectivity per neuron
  3. Participation ratio in Fourier space (how many frequencies contribute)
  4. Compare with weight-space stratum classification

If the Fourier view and stratum view agree, it validates the stratum framework
as capturing the same structure that Fourier analysis reveals, but in a more
general way that doesn't assume group structure.

Usage:
    python experiments/batch6_atlas/06_13_2026/fourier_stratum_connection.py \
        --operations multiplication,composite_addition --device cuda
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
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
from grokking_nonlinear_hunt import build_data, train_grokking_model

from transformer_lens import HookedTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

EPOCH_DEFAULTS = {
    "multiplication": 40000, "composite_addition": 15000,
    "cubing": 60000, "squaring": 40000, "polynomial": 40000,
}


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


def utc_ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def classify_stratum(singvals):
    S = np.array(singvals)
    if len(S) == 0 or S[0] < 1e-12:
        return {"stratum": "M_D", "effective_rank": 0, "gap_ratio": 0, "k_90": 0}

    total_var = np.sum(S**2)
    p_norm = S**2 / total_var
    erank = float(np.exp(-np.sum(p_norm * np.log(p_norm + 1e-30))))
    gap_ratio = float(S[0] / S[1]) if len(S) > 1 and S[1] > 1e-12 else float('inf')

    cumvar = np.cumsum(S**2) / total_var
    k_90 = int(np.searchsorted(cumvar, 0.9) + 1)
    k_90 = max(1, min(k_90, len(S)))

    if gap_ratio > 3.0 and k_90 <= 2:
        stratum = "M_1"
    elif cumvar[min(7, len(cumvar) - 1)] > 0.85 and k_90 <= 8:
        stratum = "M_k"
    elif erank < len(S) * 0.3:
        stratum = "M_Sigma"
    else:
        stratum = "M_D"

    return {"stratum": stratum, "effective_rank": round(erank, 2),
            "gap_ratio": round(gap_ratio, 4), "k_90": k_90}


def fourier_analysis(model, p):
    """Analyze Fourier structure of model weights."""
    sd = model.state_dict()
    results = {}

    W_E = sd["embed.W_E"][:p].cpu().numpy()
    F_embed = np.fft.fft(W_E, axis=0)
    F_power = np.abs(F_embed)**2
    freq_power = F_power.sum(axis=1)
    freq_power_norm = freq_power / freq_power.sum()

    n_freqs = p // 2 + 1
    freq_pr = 1.0 / np.sum(freq_power_norm[:n_freqs]**2)
    freq_pr_normalized = freq_pr / n_freqs

    top_freq_idx = np.argsort(freq_power)[::-1][:5]
    top_freqs = [(int(idx), round(float(freq_power_norm[idx]), 4)) for idx in top_freq_idx]

    results["embedding"] = {
        "fourier_pr": round(float(freq_pr), 2),
        "fourier_pr_normalized": round(float(freq_pr_normalized), 4),
        "top_frequencies": top_freqs,
        "n_freqs_50pct": int((np.cumsum(np.sort(freq_power_norm)[::-1]) < 0.5).sum()) + 1,
        "n_freqs_90pct": int((np.cumsum(np.sort(freq_power_norm)[::-1]) < 0.9).sum()) + 1,
    }

    for key in sd:
        if "W_in" in key:
            W_in = sd[key].cpu().numpy()
            n_neurons = W_in.shape[1]

            neuron_freq_selectivity = []
            for neuron_idx in range(n_neurons):
                w = W_in[:, neuron_idx]
                if np.linalg.norm(w) < 1e-10:
                    continue

                cos_scores = []
                sin_scores = []
                for freq in range(1, n_freqs):
                    basis_cos = np.cos(2 * np.pi * freq * np.arange(p) / p)
                    basis_sin = np.sin(2 * np.pi * freq * np.arange(p) / p)

                    probe_cos = W_E.T @ basis_cos
                    probe_sin = W_E.T @ basis_sin

                    cos_score = abs(np.dot(w, probe_cos)) / (np.linalg.norm(w) * np.linalg.norm(probe_cos) + 1e-10)
                    sin_score = abs(np.dot(w, probe_sin)) / (np.linalg.norm(w) * np.linalg.norm(probe_sin) + 1e-10)

                    cos_scores.append(cos_score)
                    sin_scores.append(sin_score)

                all_scores = np.array(cos_scores + sin_scores)
                if all_scores.sum() > 1e-10:
                    scores_norm = all_scores / all_scores.sum()
                    neuron_pr = 1.0 / np.sum(scores_norm**2)
                else:
                    neuron_pr = 0

                best_freq = int(np.argmax(cos_scores) + 1)
                best_score = float(max(max(cos_scores), max(sin_scores)))

                neuron_freq_selectivity.append({
                    "neuron": neuron_idx,
                    "best_freq": best_freq,
                    "best_cos_sim": round(best_score, 4),
                    "fourier_pr": round(float(neuron_pr), 2),
                    "is_fourier_selective": best_score > 0.5,
                })

            n_selective = sum(1 for n in neuron_freq_selectivity if n["is_fourier_selective"])
            avg_pr = np.mean([n["fourier_pr"] for n in neuron_freq_selectivity]) if neuron_freq_selectivity else 0

            freq_histogram = defaultdict(int)
            for n in neuron_freq_selectivity:
                if n["is_fourier_selective"]:
                    freq_histogram[n["best_freq"]] += 1

            results["mlp"] = {
                "n_neurons": n_neurons,
                "n_fourier_selective": n_selective,
                "pct_fourier_selective": round(100 * n_selective / max(n_neurons, 1), 1),
                "avg_neuron_fourier_pr": round(float(avg_pr), 2),
                "frequency_histogram": dict(sorted(freq_histogram.items())[:10]),
            }

    W_V = model.W_V[0].detach().cpu().numpy()
    W_O = model.W_O[0].detach().cpu().numpy()
    head_fourier = []
    for h in range(W_V.shape[0]):
        OV = W_V[h] @ W_O[h]
        ov_svs = np.linalg.svd(OV, compute_uv=False)
        ov_diag = classify_stratum(ov_svs)

        OV_fourier = np.fft.fft2(OV[:p, :p])
        OV_power = np.abs(OV_fourier)**2
        total_power = OV_power.sum()
        if total_power > 1e-10:
            ov_freq_norm = OV_power / total_power
            ov_freq_erank = float(np.exp(-np.sum(ov_freq_norm * np.log(ov_freq_norm + 1e-30))))
        else:
            ov_freq_erank = 0

        W_Q = model.W_Q[0, h].detach().cpu().numpy()
        W_K = model.W_K[0, h].detach().cpu().numpy()
        QK = W_Q @ W_K.T
        qk_svs = np.linalg.svd(QK, compute_uv=False)
        qk_diag = classify_stratum(qk_svs)

        head_fourier.append({
            "head": h,
            "ov_stratum": ov_diag["stratum"],
            "ov_erank": ov_diag["effective_rank"],
            "ov_fourier_erank": round(ov_freq_erank, 2),
            "qk_stratum": qk_diag["stratum"],
            "qk_erank": qk_diag["effective_rank"],
        })

    results["heads"] = head_fourier
    return results


def run_operation(operation, p, n_epochs, device, output_dir, seed):
    logger.info("=== %s (p=%d, epochs=%d) ===", operation, p, n_epochs)

    dataset, labels, train_idx, test_idx = build_data(operation, p, device)

    model, cfg, history = train_grokking_model(
        p, device, n_epochs=n_epochs,
        checkpoint_every=max(n_epochs, 1),
        dataset=dataset, labels=labels,
        train_idx=train_idx, test_idx=test_idx, seed=seed,
    )

    test_losses = history["test_losses"]
    grokked = test_losses[-1] < 0.01 if test_losses else False
    logger.info("  Final test_loss=%.4f, grokked=%s", test_losses[-1], grokked)

    logger.info("  Running Fourier analysis...")
    fourier_results = fourier_analysis(model, p)

    W_V = model.W_V[0].detach().cpu().numpy()
    W_O = model.W_O[0].detach().cpu().numpy()
    W_in = model.state_dict()["blocks.0.mlp.W_in"].cpu().numpy()
    W_out = model.state_dict()["blocks.0.mlp.W_out"].cpu().numpy()

    FF = W_in @ W_out
    ff_svs = np.linalg.svd(FF, compute_uv=False)
    ff_diag = classify_stratum(ff_svs)

    ov_strata = [h["ov_stratum"] for h in fourier_results["heads"]]
    ov_eranks = [h["ov_erank"] for h in fourier_results["heads"]]

    logger.info("  Embedding Fourier PR: %.2f (%.4f normalized), top freqs: %s",
                fourier_results["embedding"]["fourier_pr"],
                fourier_results["embedding"]["fourier_pr_normalized"],
                fourier_results["embedding"]["top_frequencies"][:3])

    if "mlp" in fourier_results:
        logger.info("  MLP: %d/%d neurons Fourier-selective (%.1f%%), avg PR=%.2f",
                    fourier_results["mlp"]["n_fourier_selective"],
                    fourier_results["mlp"]["n_neurons"],
                    fourier_results["mlp"]["pct_fourier_selective"],
                    fourier_results["mlp"]["avg_neuron_fourier_pr"])

    for h_info in fourier_results["heads"]:
        logger.info("  Head %d: OV=%s (erank=%.1f, fourier_erank=%.1f) QK=%s",
                    h_info["head"], h_info["ov_stratum"], h_info["ov_erank"],
                    h_info["ov_fourier_erank"], h_info["qk_stratum"])

    logger.info("  FF circuit: %s (erank=%.1f)", ff_diag["stratum"], ff_diag["effective_rank"])

    rec = {
        "timestamp": utc_ts(),
        "operation": operation, "p": p, "seed": seed,
        "grokked": grokked,
        "final_test_loss": round(test_losses[-1], 6),
        "fourier": fourier_results,
        "ff_stratum": ff_diag["stratum"],
        "ff_erank": ff_diag["effective_rank"],
        "agreement": {
            "embedding_fourier_pr": fourier_results["embedding"]["fourier_pr_normalized"],
            "mlp_fourier_selective_pct": fourier_results.get("mlp", {}).get("pct_fourier_selective", 0),
            "head_ov_strata": ov_strata,
            "head_ov_eranks": ov_eranks,
        },
    }

    out_path = Path(output_dir) / f"fourier_stratum_{operation}.jsonl"
    with open(out_path, "a") as f:
        f.write(json.dumps(rec, cls=NumpyEncoder) + "\n")
    logger.info("  Results written to %s", out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--operations", default="multiplication,composite_addition,cubing")
    parser.add_argument("--p", type=int, default=113)
    parser.add_argument("--n-epochs", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="experiments/results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    operations = [op.strip() for op in args.operations.split(",")]

    for op in operations:
        n_ep = args.n_epochs if args.n_epochs > 0 else EPOCH_DEFAULTS.get(op, 25000)
        run_operation(op, args.p, n_ep, args.device, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
