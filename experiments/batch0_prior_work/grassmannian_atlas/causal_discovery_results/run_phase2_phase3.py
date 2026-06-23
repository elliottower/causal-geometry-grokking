"""Run Phase 2 (causal discovery) + Phase 3 (mechval instruments) on L11 interventional data.

Handles the 400/200 row mismatch: Phase 1 collected activations on all 400 examples
(200 IOI + 200 SVA) but logit diffs only on 200 per task. We split and run separately,
then merge results.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from causal_discovery_grassmannian import (
    _run_pc,
    _run_ges,
    _run_notears,
    _run_dcdi,
    _run_icp,
    _run_igsp,
    _build_intervention_targets,
    run_mechval_instruments,
)


def build_observation_matrix_fixed(collected):
    """Build observation matrix handling the 400/200 split correctly.

    The Phase 1 data has:
      dir_activations_{a,b}: 400 rows (200 IOI examples + 200 SVA examples)
      task_a_logit_diff: 200 (IOI only)
      task_b_logit_diff: 200 (SVA only)

    We build two observation matrices:
      IOI view: first 200 activation rows + task_a_logit_diff
      SVA view: last 200 activation rows + task_b_logit_diff
    Then stack them for a combined matrix, using NaN for the missing logit diff.
    """
    baseline = collected["baseline"]
    k = collected["meta"]["das_k"]

    dir_acts_a = np.array(baseline["dir_activations_a"])  # (400, k)
    dir_acts_b = np.array(baseline["dir_activations_b"])  # (400, k)
    ld_a = np.array(baseline["task_a_logit_diff"])  # (200,)
    ld_b = np.array(baseline["task_b_logit_diff"])  # (200,)

    n_a = len(ld_a)
    n_b = len(ld_b)

    # IOI examples: rows 0..199
    ioi_acts_a = dir_acts_a[:n_a]  # (200, k)
    ioi_acts_b = dir_acts_b[:n_a]  # (200, k)

    # SVA examples: rows 200..399
    sva_acts_a = dir_acts_a[n_a:n_a + n_b]  # (200, k)
    sva_acts_b = dir_acts_b[n_a:n_a + n_b]  # (200, k)

    # Combined: 400 rows, with NaN for missing logit diff
    ld_a_full = np.concatenate([ld_a, np.full(n_b, np.nan)])
    ld_b_full = np.concatenate([np.full(n_a, np.nan), ld_b])

    obs_combined = np.hstack([dir_acts_a, dir_acts_b,
                              ld_a_full.reshape(-1, 1),
                              ld_b_full.reshape(-1, 1)])

    # For algorithms that can't handle NaN: use IOI-only (200 rows, task_a output)
    obs_ioi = np.hstack([ioi_acts_a, ioi_acts_b, ld_a.reshape(-1, 1)])
    obs_sva = np.hstack([sva_acts_a, sva_acts_b, ld_b.reshape(-1, 1)])

    var_names = (
        [f"canon_dir_a_{i}" for i in range(k)]
        + [f"canon_dir_b_{i}" for i in range(k)]
        + ["logit_diff_a", "logit_diff_b"]
    )

    var_names_single = (
        [f"canon_dir_a_{i}" for i in range(k)]
        + [f"canon_dir_b_{i}" for i in range(k)]
        + ["logit_diff"]
    )

    return {
        "combined": (obs_combined, var_names),
        "ioi": (obs_ioi, var_names_single),
        "sva": (obs_sva, var_names_single),
    }


def run_phase2(collected, methods=None):
    """Run all causal discovery algorithms with proper data handling."""
    if methods is None:
        methods = ["notears", "pc", "ges", "dcdi", "icp", "igsp"]

    matrices = build_observation_matrix_fixed(collected)
    k = collected["meta"]["das_k"]
    results = {}

    for method in methods:
        print(f"\n  --- {method.upper()} ---")
        t0 = time.time()
        try:
            if method == "notears":
                # Use IOI-only matrix (no NaN)
                obs, var_names = matrices["ioi"]
                print(f"    Using IOI matrix: {obs.shape}")
                dag = _run_notears(obs, var_names)

            elif method == "pc":
                obs, var_names = matrices["ioi"]
                print(f"    Using IOI matrix: {obs.shape}")
                dag = _run_pc(obs, var_names)

            elif method == "ges":
                obs, var_names = matrices["ioi"]
                print(f"    Using IOI matrix: {obs.shape}")
                dag = _run_ges(obs, var_names)

            elif method == "dcdi":
                obs_combined, var_names = matrices["combined"]
                # DCDI uses intervention targets, needs the combined var_names
                intervention_targets = _build_intervention_targets(collected, var_names)
                # But its internal logic handles the missing values
                obs_ioi, _ = matrices["ioi"]
                # Use IOI matrix for obs, but pass full intervention targets
                dag = _run_dcdi_fixed(obs_ioi, var_names[:len(var_names)-1],
                                      intervention_targets, collected)

            elif method == "icp":
                obs, var_names = matrices["ioi"]
                print(f"    Using IOI matrix: {obs.shape}")
                dag = _run_icp(obs, var_names, collected)

            elif method == "igsp":
                obs, var_names = matrices["ioi"]
                intervention_targets = _build_intervention_targets(
                    collected, var_names + ["logit_diff_b"])
                dag = _run_igsp(obs, var_names, intervention_targets, collected)

            else:
                print(f"    Unknown: {method}")
                continue

            dt = time.time() - t0
            results[method] = {
                "adjacency": dag["adjacency"],
                "var_names": dag.get("var_names", var_names) if isinstance(dag.get("var_names"), list) else var_names,
                "n_edges": dag.get("n_edges", 0),
                "runtime_s": dt,
                "extra": dag.get("extra", {}),
            }
            print(f"    Found {results[method]['n_edges']} edges in {dt:.1f}s")

        except ImportError as e:
            print(f"    SKIPPED (missing dependency): {e}")
            results[method] = {"error": str(e)}
        except Exception as e:
            import traceback
            print(f"    FAILED: {e}")
            traceback.print_exc()
            results[method] = {"error": str(e)}

    return results


def _run_dcdi_fixed(obs, var_names, intervention_targets, collected):
    """DCDI adapted for single-task observation matrix."""
    n, d = obs.shape
    obs_std = (obs - obs.mean(0)) / (obs.std(0) + 1e-8)
    edge_scores = np.zeros((d, d))

    idx_ld = d - 1  # logit_diff is last column
    baseline_a = np.array(collected["baseline"]["task_a_logit_diff"])

    for target in intervention_targets:
        if target["type"] != "zero":
            continue
        var_idx = target["var_idx"]
        if var_idx >= d:
            continue
        outcomes_a = np.array(target["task_a_outcomes"][:len(baseline_a)])
        if len(outcomes_a) > 0:
            delta = np.mean(np.abs(outcomes_a - baseline_a[:len(outcomes_a)]))
            edge_scores[var_idx, idx_ld] = delta

    threshold = np.percentile(edge_scores[edge_scores > 0], 25) if (edge_scores > 0).any() else 0
    adj = (edge_scores > threshold).astype(float)
    np.fill_diagonal(adj, 0)

    return {
        "adjacency": adj.tolist(),
        "n_edges": int(adj.sum()),
        "extra": {"threshold": float(threshold), "method": "dcdi_simplified_fixed"},
    }


def run_phase3_fixed(collected):
    """Run Phase 3 with patched baseline to handle 400/200 split."""
    # Patch baseline: truncate activations to match logit diffs
    patched = json.loads(json.dumps(collected))
    n_a = len(patched["baseline"]["task_a_logit_diff"])
    patched["baseline"]["dir_activations_a"] = patched["baseline"]["dir_activations_a"][:n_a]
    patched["baseline"]["dir_activations_b"] = patched["baseline"]["dir_activations_b"][:n_a]

    return run_mechval_instruments(
        Path(__file__).parent, type("Args", (), {"layer": 11})
    )


def print_summary(phase2_results, phase3_results):
    """Print interpretable summary of all results."""
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)

    k = 32

    # Phase 2: Causal Discovery
    print("\n  PHASE 2: Causal Discovery DAGs")
    print("  " + "-" * 50)
    for method, res in phase2_results.items():
        if "error" in res:
            print(f"    {method.upper():10s}: ERROR — {res['error'][:60]}")
        else:
            n_edges = res["n_edges"]
            rt = res.get("runtime_s", 0)
            print(f"    {method.upper():10s}: {n_edges:3d} edges ({rt:.1f}s)")

            # Show which variables have most edges
            adj = np.array(res["adjacency"])
            var_names = res.get("var_names", [])
            out_degree = (adj != 0).sum(axis=1)
            in_degree = (adj != 0).sum(axis=0)
            top_out = np.argsort(out_degree)[::-1][:5]
            for idx in top_out:
                if out_degree[idx] > 0 and idx < len(var_names):
                    print(f"              {var_names[idx]}: out={int(out_degree[idx])}, in={int(in_degree[idx])}")

    # Phase 3: Mechval Instruments
    if phase3_results:
        print("\n  PHASE 3: Mechval Causal Instruments")
        print("  " + "-" * 50)

        # Pearl
        pearl = phase3_results.get("scm_pearl", {}).get("per_direction", [])
        if pearl:
            causal_dirs = [(i, p["effect_a"], p["effect_b"]) for i, p in enumerate(pearl)]
            causal_dirs.sort(key=lambda x: x[1], reverse=True)
            print(f"    Pearl (activation patching): top 5 causal directions for IOI:")
            for i, ea, eb in causal_dirs[:5]:
                selectivity = ea / (eb + 1e-8)
                print(f"      dir {i:2d}: IOI effect={ea:.3f}, SVA effect={eb:.3f}, selectivity={selectivity:.1f}x")

        # Woodward
        woodward = phase3_results.get("woodward", {}).get("per_direction", [])
        if woodward:
            n_monotonic = sum(1 for w in woodward if w.get("is_monotonic"))
            n_graded = sum(1 for w in woodward if w.get("graded"))
            print(f"    Woodward (dose-response): {n_monotonic}/{len(woodward)} monotonic, {n_graded}/{len(woodward)} graded")

        # Counterfactual
        cf = phase3_results.get("counterfactual", {}).get("per_direction", [])
        if cf:
            effects = [c.get("swap_effect_a", 0) for c in cf]
            print(f"    Counterfactual (swap): mean effect={np.mean(effects):.3f}, max={max(effects):.3f}")

        # Synergy
        mdc = phase3_results.get("mdc", {})
        if "n_additive" in mdc:
            print(f"    Composition: {mdc['n_additive']} additive, {mdc['n_non_additive']} non-additive pairs")
            print(f"      mean |synergy|={mdc.get('mean_abs_synergy', 0):.3f}")

        # Transportability
        transport = phase3_results.get("transportability", {})
        if "cross_task_correlation" in transport:
            print(f"    Transportability: cross-task effect correlation = {transport['cross_task_correlation']:.3f}")

        # Grassmann Woodward
        gw = phase3_results.get("grassmann_woodward", {})
        if gw:
            print(f"    Grassmannian Woodward: crossover={gw.get('has_crossover')}, "
                  f"A monotonic={gw.get('a_monotonic')}, B monotonic={gw.get('b_monotonic')}")

        # Subspace selectivity
        ss = phase3_results.get("subspace_selectivity", {})
        if ss:
            print(f"    Subspace selectivity: remove A → A loss={ss.get('remove_a_effect_a', 0):.3f}, "
                  f"B spared={ss.get('remove_a_effect_b', 0):.3f}")

        # Verdicts
        verdicts = phase3_results.get("verdicts", {})
        if verdicts:
            print(f"\n    Per-direction verdicts:")
            for i, v in enumerate(verdicts[:10]):
                if isinstance(v, dict):
                    label = v.get("verdict", "?")
                    score = v.get("score", 0)
                    print(f"      dir {i:2d}: {label} (score={score:.2f})")


def main():
    data_path = Path(__file__).parent / "interventional_data.json"
    print(f"Loading data from {data_path}")
    with open(data_path) as f:
        collected = json.load(f)

    meta = collected["meta"]
    print(f"Checkpoint: {meta['checkpoint']}")
    print(f"Tasks: {meta['tasks']}, Layer: {meta['layer']}, k={meta['das_k']}")
    print(f"d_model: {meta['d_model']}, n_factors: {meta.get('n_factors', 'N/A')}")
    print(f"Canonical angles: {[f'{a:.2f}' for a in meta.get('canonical_angles_rad', [])[:5]]} ...")

    # Phase 2
    print(f"\n{'='*70}")
    print(f"  PHASE 2: Causal Discovery")
    print(f"{'='*70}")
    phase2_results = run_phase2(collected)

    # Phase 3 (patched for data mismatch)
    print(f"\n{'='*70}")
    print(f"  PHASE 3: Mechval Causal Instruments")
    print(f"{'='*70}")

    # Patch baseline for Phase 3
    n_a = len(collected["baseline"]["task_a_logit_diff"])
    collected["baseline"]["dir_activations_a"] = collected["baseline"]["dir_activations_a"][:n_a]
    collected["baseline"]["dir_activations_b"] = collected["baseline"]["dir_activations_b"][:n_a]

    phase3_results = run_mechval_instruments(
        str(Path(__file__).parent),
        type("Args", (), {"layer": 11})()
    )

    # Save results
    output = {
        "meta": meta,
        "phase2_causal_discovery": phase2_results,
        "phase3_mechval_instruments": phase3_results,
        "data_note": "L11 interventional data uses dense-k32-grassmann DAS on atomic-sweep-40 (source mismatch)",
    }
    out_path = Path(__file__).parent / "phase2_phase3_results_L11.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    print_summary(phase2_results, phase3_results)


if __name__ == "__main__":
    main()
