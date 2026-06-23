"""Published baseline numbers from the mechanistic interpretability literature.

Reference scores for comparing factorized model circuit discovery against
established methods. Every value has a Citation with a URL and exact
table/figure/section location so it can be independently verified.

For MIB leaderboard raw data (per-circuit-size faithfulness curves),
see ``reference/mib-leaderboard/eval-results-mib-subgraph/baselines/``.
That directory is a git submodule of the HuggingFace space
``mib-bench/leaderboard`` and can be updated with ``git submodule update``.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Citation:
    paper: str
    venue: str
    arxiv_id: str
    location: str

    @property
    def pdf_url(self) -> str:
        if not self.arxiv_id:
            return ""
        return f"https://arxiv.org/pdf/{self.arxiv_id}"

    @property
    def html_url(self) -> str:
        if not self.arxiv_id:
            return ""
        return f"https://arxiv.org/html/{self.arxiv_id}"

    def __repr__(self) -> str:
        return f"{self.paper} ({self.venue}) [{self.location}] {self.pdf_url}"


# ---- Paper citations -------------------------------------------------------

MIB = Citation(
    paper="Mueller et al., MIB: A Mechanistic Interpretability Benchmark",
    venue="ICML 2025",
    arxiv_id="2504.13151v2",
    location="Table 2 (CMD), Table 3 (IIA/MSE)",
)
MIB_PMLR = "https://proceedings.mlr.press/v267/mueller25a.html"

WANG_IOI = Citation(
    paper="Wang et al., Interpretability in the Wild: IOI in GPT-2 Small",
    venue="ICLR 2023",
    arxiv_id="2211.00593v1",
    location="Section 2 (task description) + Section 4 (faithfulness)",
)

HANNA_GT = Citation(
    paper="Hanna et al., How does GPT-2 compute greater-than?",
    venue="NeurIPS 2023",
    arxiv_id="2305.00586v2",
    location="Section 2 (quantitative eval) + Section 3.2 (circuit eval)",
)

CONMY_ACDC = Citation(
    paper="Conmy et al., Towards Automated Circuit Discovery",
    venue="NeurIPS 2023",
    arxiv_id="2304.14997v3",
    location="Table 2 (Appendix D, AUC) + Abstract (68/32k edges)",
)

LAZO_SVA = Citation(
    paper="Lazo et al., Identifying a Circuit for Verb Conjugation in GPT-2",
    venue="arXiv 2025",
    arxiv_id="2506.22105v1",
    location="Table 3 (Section 4.2) + Figure 1",
)

WARSTADT_BLIMP = Citation(
    paper="Warstadt et al., BLiMP: Benchmark of Linguistic Minimal Pairs",
    venue="TACL 2020",
    arxiv_id="1912.00582v4",
    location="Table 3 (category-level); per-paradigm from GitHub supplementary",
)
BLIMP_SUPP = "https://github.com/alexwarstadt/blimp/blob/master/raw_results/summary/models_summary.jsonl"

CDT_GT = Citation(
    paper="Garcia-Carrasco et al., CD-T: Circuit Discovery for Transformers",
    venue="ICLR 2025",
    arxiv_id="2407.00886",
    location="Table 1 (task-specific metrics)",
)

MARKS_SFC = Citation(
    paper="Marks et al., Sparse Feature Circuits",
    venue="ICLR 2025",
    arxiv_id="2403.19647",
    location="Section 5 (SVA case study, Pythia-70M not GPT-2)",
)

MATHWIN_GENDER = Citation(
    paper="Mathwin, Preliminary Circuit for Gendered Pronouns in GPT-2 Small",
    venue="MATS Hackathon 2023 (unpublished)",
    arxiv_id="",
    location="hackathon report; ACDC-discovered circuit >= full model",
)

OLSSON_INDUCTION = Citation(
    paper="Olsson et al., In-context Learning and Induction Heads",
    venue="arXiv 2022",
    arxiv_id="2209.11895",
    location="Section 2 (mechanism) + Table 5 Appendix (GPT-2 Small head list)",
)


# ---------------------------------------------------------------------------
# MIB Circuit Localization — Track 1
#
# Metric: CMD (Circuit-Model Distance), lower is better.
# CMD = integral of |1 - faithfulness(k)| over circuit size fractions
#       (Equation 2, Section 3.1).
# Faithfulness f = [m(circuit) - m(empty)] / [m(full) - m(empty)]
#       (Equation 1, Section 3.1).
# Evaluated at k in {0.1%, 0.2%, 0.5%, 1%, 2%, 5%, 10%, 20%, 50%, 100%}.
#
# Source: Table 2 of PMLR published version (mueller25a).
# Verified against: proceedings.mlr.press/v267/mueller25a.html
# Raw per-circuit-size faithfulness: reference/mib-leaderboard/
#
# IB column = InterpBench AUROC (synthetic ground-truth circuit).
# Dashes = method not evaluated on that model/task.
# ---------------------------------------------------------------------------

MIB_CIRCUIT_CMD: dict[str, dict[str, float | None]] = {
    # fmt: off
    # ---- GPT-2 Small (124M, 12L) — IOI only ----
    "gpt2_ioi": {
        "EActP_CF":         0.02,
        "EAP_CF":           0.03,
        "EAP_IG_inp_CF":    0.03,
        "EAP_IG_act_CF":    0.03,
        "UGS":              0.03,
        "NAP_IG_CF":        0.27,
        "EAP_mean":         0.29,
        "EAP_OA":           0.30,
        "NAP_CF":           0.38,
        "IFR":              0.42,
        "random":           0.75,
    },
    # ---- Qwen-2.5 (0.5B) ----
    "qwen25_ioi": {
        "EAP_IG_act_CF": 0.01,  "EAP_IG_inp_CF": 0.02,  "UGS": 0.03,
        "EAP_CF": 0.15,  "EAP_OA": 0.16,  "EAP_mean": 0.18,
        "NAP_IG_CF": 0.20,  "NAP_CF": 0.33,  "EActP_CF": 0.49,
        "IFR": 0.69,  "random": 0.72,
    },
    "qwen25_mcqa": {
        "EAP_IG_act_CF": 0.05,  "EAP_CF": 0.07,  "EAP_IG_inp_CF": 0.08,
        "EAP_OA": 0.11,  "NAP_IG_CF": 0.18,  "UGS": 0.20,
        "EAP_mean": 0.21,  "NAP_CF": 0.30,  "EActP_CF": 0.36,
        "IFR": 0.60,  "random": 0.73,
    },
    # ---- Gemma-2 (2B) ----
    "gemma2_ioi": {
        "EAP_IG_act_CF": 0.03,  "EAP_IG_inp_CF": 0.04,
        "EAP_CF": 0.06,  "EAP_mean": 0.25,  "NAP_IG_CF": 0.26,
        "NAP_CF": 0.37,  "random": 0.69,  "IFR": 0.75,
    },
    "gemma2_mcqa": {
        "EAP_IG_act_CF": 0.07,  "EAP_IG_inp_CF": 0.06,
        "EAP_CF": 0.08,  "EAP_mean": 0.20,  "NAP_IG_CF": 0.29,
        "NAP_CF": 0.35,  "IFR": 0.62,  "random": 0.68,
    },
    "gemma2_arc_easy": {
        "EAP_IG_act_CF": 0.04,  "EAP_IG_inp_CF": 0.04,
        "EAP_CF": 0.04,  "EAP_mean": 0.22,  "NAP_IG_CF": 0.28,
        "NAP_CF": 0.33,  "IFR": 0.66,  "random": 0.68,
    },
    # ---- Llama-3.1 (8B) ----
    "llama31_ioi": {
        "EAP_IG_act_CF": 0.01,  "EAP_IG_inp_CF": 0.01,
        "EAP_CF": 0.01,  "EAP_mean": 0.04,  "NAP_IG_CF": 0.19,
        "NAP_CF": 0.29,  "random": 0.74,  "IFR": 0.83,
    },
    "llama31_arith": {
        "EAP_IG_act_CF": 0.00,  "EAP_IG_inp_CF": 0.00,
        "EAP_CF": 0.01,  "EAP_mean": 0.07,  "NAP_IG_CF": 0.18,
        "IFR": 0.22,  "NAP_CF": 0.28,  "random": 0.75,
    },
    "llama31_mcqa": {
        "EAP_IG_act_CF": 0.13,  "EAP_IG_inp_CF": 0.14,
        "EAP_CF": 0.09,  "EAP_mean": 0.16,  "NAP_IG_CF": 0.33,
        "NAP_CF": 0.32,  "IFR": 0.48,  "random": 0.74,
    },
    "llama31_arc_easy": {
        "EAP_IG_inp_CF": 0.11,  "EAP_CF": 0.11,
        "EAP_mean": 0.28,  "EAP_IG_act_CF": 0.30,
        "NAP_IG_CF": 0.67,  "IFR": 0.64,  "NAP_CF": 0.69,  "random": 0.74,
    },
    "llama31_arc_challenge": {
        "EAP_CF": 0.18,  "EAP_mean": 0.20,  "EAP_IG_inp_CF": 0.22,
        "EAP_IG_act_CF": 0.37,  "NAP_IG_CF": 0.67,
        "NAP_CF": 0.69,  "random": 0.74,  "IFR": 0.76,
    },
    # fmt: on
}
MIB_CIRCUIT_CMD_CITE = MIB  # Table 2

# ---------------------------------------------------------------------------
# MIB Causal Variable Localization — Track 2
#
# Metric: IIA (Interchange Intervention Accuracy), higher is better.
# Exception: IOI/GPT-2 uses MSE (lower is better).
# Source: Table 3 of PMLR published version.
# Verified correct by Perplexity audit against PMLR HTML.
# ---------------------------------------------------------------------------

MIB_CAUSAL_IIA: dict[str, dict[str, dict[str, float]]] = {
    "mcqa": {
        "gemma2": {"DAS_O_Answer": 95, "DAS_X_Order": 77, "DBM": 84, "DBM_PCA": 57, "DBM_SAE": 73, "full_vector": 61},
        "llama31": {"DAS_O_Answer": 94, "DAS_X_Order": 77, "DBM": 86, "DBM_PCA": 65, "DBM_SAE": 80, "full_vector": 77},
        "qwen25": {"DAS_O_Answer": 86, "DAS_X_Order": 78, "DBM": 46, "DBM_PCA": 22, "full_vector": 35},
    },
    "arc_easy": {
        "gemma2": {"DAS_O_Answer": 88, "DAS_X_Order": 76, "DBM": 82, "DBM_PCA": 78, "DBM_SAE": 70, "full_vector": 63},
        "llama31": {"DAS_O_Answer": 88, "DAS_X_Order": 74, "DBM": 85, "DBM_PCA": 84, "DBM_SAE": 74, "full_vector": 68},
    },
}

MIB_CAUSAL_IOI_MSE: dict[str, dict[str, float] | str] = {
    "_note": "GPT-2 IOI uses MSE (lower=better), not IIA — different metric from all other Table 3 tasks",
    # arxiv v2 / PMLR camera-ready values (v1 had different numbers: 1.93/2.19 etc.)
    "gpt2": {
        "DAS_S_Pos": 2.20, "DAS_S_Tok": 2.08,
        "DBM_S_Pos": 2.22, "DBM_S_Tok": 2.35,
        "DBM_PCA_S_Pos": 2.24, "DBM_PCA_S_Tok": 2.33,
        "full_vector_S_Pos": 2.45, "full_vector_S_Tok": 2.82,
    },
}
MIB_CAUSAL_CITE = Citation(
    paper=MIB.paper, venue=MIB.venue, arxiv_id=MIB.arxiv_id,
    location="Table 3 sub-table (d)/(f) for IOI MSE; rest of Table 3 for IIA. Values from v2/PMLR (v1 differs).",
)

# ---------------------------------------------------------------------------
# ACDC Circuit Discovery (Conmy et al. 2023)
#
# Metric: AUC over ROC of circuit edges, higher is better.
# Source: Table 2, Appendix D (KL metric, Edge-level).
# Corrected from paper — previous values (0.853, 0.806, 0.693) were wrong.
# ---------------------------------------------------------------------------

ACDC_AUC: dict[str, dict[str, float]] = {
    # Edge-level, KL divergence metric
    "greater_than_kl_edge": {"ACDC": 0.883, "SP": 0.820, "HISP": 0.279},
    "ioi_kl_edge": {"ACDC": 0.868, "SP": 0.888, "HISP": 0.239},
    "docstring_kl_edge": {"ACDC": 0.434, "SP": 0.937, "HISP": 0.183},
    # Edge-level, task-specific metric
    "greater_than_task_edge": {"ACDC": 0.461, "SP": 0.848, "HISP": 0.275},
    "ioi_task_edge": {"ACDC": 0.589, "SP": 0.837, "HISP": 0.227},
    "docstring_task_edge": {"ACDC": 0.972, "SP": 0.942, "HISP": 0.177},
    "tracr_xproportion_task_edge": {"ACDC": 0.600, "SP": 0.400, "HISP": 0.679},
    "tracr_reverse_task_edge": {"ACDC": 0.200, "SP": 0.416, "HISP": 0.656},
}
ACDC_AUC_CITE = Citation(
    paper=CONMY_ACDC.paper, venue=CONMY_ACDC.venue,
    arxiv_id=CONMY_ACDC.arxiv_id,
    location="Table 2, Appendix D (Edge-level rows; KL and Task-specific metrics)",
)

ACDC_EDGES: dict[str, dict[str, int]] = {
    "greater_than": {"ACDC": 68, "total": 32_000},
}
ACDC_EDGES_CITE = Citation(
    paper=CONMY_ACDC.paper, venue=CONMY_ACDC.venue,
    arxiv_id=CONMY_ACDC.arxiv_id, location="Abstract",
)

# ---------------------------------------------------------------------------
# Task-level reference scores — GPT-2 Small full model
# ---------------------------------------------------------------------------

GPT2_FULL_MODEL: dict[str, dict[str, float | str]] = {
    "ioi": {
        "logit_diff": 3.56,
        "io_over_s_pct": 99.3,
        "io_prob": 0.49,
        "_cite": "Wang et al. Section 2, 'Task description' paragraph, single sentence",
        "_url": "https://arxiv.org/pdf/2211.00593",
    },
    "greater_than": {
        "prob_diff": 0.817,
        "prob_diff_sd": 0.193,
        "cutoff_sharpness": 0.060,
        "_cite": "Hanna et al. Section 2, 'Quantitative Evaluation' subsection, inline",
        "_url": "https://arxiv.org/pdf/2305.00586",
    },
    "sva_base": {
        "logit_diff_acc": 0.70,
        "_cite": "Lazo et al. Table 3 (Section 4.2), 'Base' row, 'Full Model Accuracy' col",
        "_url": "https://arxiv.org/pdf/2506.22105",
    },
}

# BLiMP behavioral baselines.
# IMPORTANT: BLiMP paper v4 uses GPT-2 Large (774M, 36L), NOT GPT-2 Small.
# Category-level from paper Table 3. Per-paradigm from GitHub supplementary.
BLIMP_GPT2: dict[str, dict[str, float | str]] = {
    "_note": "BLiMP v4 (Feb 2023) changed to GPT-2 Large (774M), not Small (124M)",
    "anaphor_gender_agreement": {
        "minimal_pair_acc": 0.994,
        "_cite": "BLiMP GitHub supplementary (models_summary.jsonl)",
        "_url": BLIMP_SUPP,
    },
    "anaphor_number_agreement": {
        "minimal_pair_acc": 0.992,
        "_cite": "BLiMP GitHub supplementary (models_summary.jsonl)",
        "_url": BLIMP_SUPP,
    },
    "regular_plural_sva_1": {
        "minimal_pair_acc": 0.967,
        "_cite": "BLiMP GitHub supplementary (models_summary.jsonl)",
        "_url": BLIMP_SUPP,
    },
    "regular_plural_sva_2": {
        "minimal_pair_acc": 0.910,
        "_cite": "BLiMP GitHub supplementary (models_summary.jsonl)",
        "_url": BLIMP_SUPP,
    },
    "sva_category": {
        "category_acc": 0.864,
        "_cite": "Warstadt et al. Table 3 (Section 5)",
        "_url": "https://arxiv.org/pdf/1912.00582",
    },
    "anaphor_category": {
        "category_acc": 0.993,
        "_cite": "Warstadt et al. Table 3 (Section 5)",
        "_url": "https://arxiv.org/pdf/1912.00582",
    },
    "overall": {
        "acc": 0.830,
        "_cite": "Warstadt et al. Table 3 (Section 5)",
        "_url": "https://arxiv.org/pdf/1912.00582",
    },
}

# ---------------------------------------------------------------------------
# Best published circuit recovery — GPT-2 Small
# ---------------------------------------------------------------------------

CIRCUIT_RECOVERY: dict[str, dict[str, float | int | str]] = {
    "ioi": {
        "metric": "logit_diff",
        "full_model_score": 3.56,
        "faithfulness_gap": 0.46,
        "recovery_pct": 87.0,
        "n_heads": 26,
        "_cite": "Wang et al. Section 4 para 1: '|F(M)-F(C)|=0.46, or only 13% of F(M)=3.56'",
        "_url": "https://arxiv.org/pdf/2211.00593",
        "_note": "circuit_score 3.10 is derived (3.56-0.46), not stated in paper",
    },
    "greater_than": {
        "metric": "prob_diff",
        "circuit_score": 0.727,
        "full_model_score": 0.817,
        "recovery_pct": 89.5,
        "n_attn_heads": 7,
        "n_mlps": 4,
        "necessity_score": -0.366,
        "_cite": "Hanna et al. Section 3.2 'Evaluation' subsection, inline text",
        "_url": "https://arxiv.org/pdf/2305.00586",
    },
    "greater_than_generalization": {
        "started_ended_recovery": 0.988,
        "luxury_goods_recovery": 0.889,
        "number_sequences_recovery": 0.903,
        "_cite": "Hanna et al. Section 5, inline text + Appendix F",
        "_url": "https://arxiv.org/pdf/2305.00586",
    },
    "sva_base": {
        "metric": "logit_diff_acc",
        "circuit_score": 0.65,
        "full_model_score": 0.70,
        "recovery_pct": 93.0,
        "n_heads": 12,
        "_cite": "Lazo et al. Table 3 (Section 4.2), 'Base' row",
        "_url": "https://arxiv.org/pdf/2506.22105",
    },
    "sva_complex": {
        "metric": "logit_diff_acc",
        "circuit_score": 0.80,
        "full_model_score": 1.00,
        "recovery_pct": 80.0,
        "n_heads": 125,
        "_cite": "Lazo et al. Table 3 (Section 4.2), 'Complex' row",
        "_url": "https://arxiv.org/pdf/2506.22105",
    },
    "gendered_pronoun": {
        "metric": "logit_diff_he_she",
        "recovery_pct": 100.0,
        "_cite": "Mathwin 2023 MATS hackathon report (unpublished, not peer-reviewed)",
        "_note": "ACDC circuit >= full model performance",
    },
}

# ---------------------------------------------------------------------------
# Known circuit components — GPT-2 Small (layer, head) tuples
# ---------------------------------------------------------------------------

KNOWN_CIRCUITS: dict[str, dict[str, list[tuple[int, ...]] | str]] = {
    "ioi": {
        "name_movers": [(9, 9), (10, 0), (9, 6)],
        "negative_name_movers": [(10, 7), (11, 10)],
        "_negative_name_movers_note": "Copy suppression heads — self-contained, no upstream composition edges",
        "s_inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
        "duplicate_token": [(0, 1), (3, 0)],
        "induction": [(5, 5), (6, 9)],
        "previous_token": [(2, 2), (4, 11)],
        "backup_name_movers": [(9, 0), (9, 7), (10, 1), (10, 2), (10, 6), (10, 10), (11, 2), (11, 9)],
        "_backup_name_movers_cite": "Wang et al. Appendix B + Section 4",
        "_cite": "Wang et al. Figure 2 + Section 3 (all head assignments)",
        "_url": "https://arxiv.org/pdf/2211.00593",
    },
    "greater_than": {
        "attention": [(5, 1), (5, 5), (6, 9), (7, 10), (8, 8), (8, 11), (9, 1)],
        "mlps": [(8,), (9,), (10,), (11,)],
        "_cite": "Hanna et al. Section 3.2 'Attention Heads' subsection, inline",
        "_url": "https://arxiv.org/pdf/2305.00586",
    },
    "sva_base": {
        "top_12": [(11, 7), (11, 6), (0, 4), (11, 4), (0, 8), (2, 6), (1, 0), (2, 1), (1, 1), (6, 0), (10, 0), (9, 4)],
        "_cite": "Lazo et al. Section 4.1 + Figure 1 (head 11.7 most influential)",
        "_url": "https://arxiv.org/pdf/2506.22105",
    },
    "induction": {
        "induction": [(5, 1), (5, 5), (6, 9), (7, 2), (7, 10)],
        "previous_token": [(2, 2), (4, 11)],
        "_cite": "Olsson et al. Table 5 Appendix (GPT-2 Small heads)",
        "_url": "https://arxiv.org/pdf/2209.11895",
    },
    "gendered_pronoun": {
        "gender_attribute": [(0, 10), (3, 0), (5, 8)],
        "name_binding": [(6, 6), (8, 6)],
        "_cite": "Mathwin 2023 (MATS hackathon, unpublished)",
    },
}


# ---------------------------------------------------------------------------
# MIB Leaderboard CMD — from HuggingFace space (different from paper!)
#
# The leaderboard uses FIXED x-axis percentages
#   (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
# while the paper uses SELF-NORMALIZED x-axis (edge_counts / max_edges)
# and averages over 3 seeds. This causes significant differences.
#
# Source: reference/mib-leaderboard/eval-results-mib-subgraph/baselines/*.json
# These are the values the online leaderboard actually displays.
# ---------------------------------------------------------------------------

MIB_LEADERBOARD_CMD: dict[str, dict[str, float]] = {
    "EAP_CF": {
        "ioi_gpt2": 0.214, "ioi_qwen25": 0.736, "ioi_gemma2": 0.297,
        "mcqa_qwen25": 0.247, "mcqa_gemma2": 0.355, "mcqa_llama3": 0.235,
        "arith_add_llama3": 0.462, "arith_sub_llama3": 0.324,
        "arc_easy_gemma2": 0.190, "arc_challenge_llama3": 0.743,
    },
    "EAP_OA": {
        "ioi_gpt2": 0.240, "ioi_qwen25": 0.299,
        "mcqa_qwen25": 0.675,
    },
    "EAP_mean": {
        "ioi_gpt2": 0.713, "ioi_qwen25": 0.298, "ioi_gemma2": 0.320,
        "mcqa_qwen25": 0.692, "mcqa_gemma2": 0.861, "mcqa_llama3": 0.667,
        "arith_add_llama3": 0.529, "arith_sub_llama3": 0.667,
        "arc_easy_gemma2": 0.676, "arc_easy_llama3": 0.655,
        "arc_challenge_llama3": 0.697,
    },
    "EAP_IG_act_CF": {
        "ioi_gpt2": 0.677, "ioi_qwen25": 0.598, "ioi_gemma2": 0.286,
        "ioi_llama3": 0.067,
        "mcqa_qwen25": 0.223, "mcqa_gemma2": 0.396, "mcqa_llama3": 0.843,
        "arith_add_llama3": 0.522, "arith_sub_llama3": 0.032,
        "arc_easy_gemma2": 0.624, "arc_easy_llama3": 0.749,
        "arc_challenge_llama3": 0.736,
    },
    "EAP_IG_inp_CF": {
        "ioi_gpt2": 1.016, "ioi_qwen25": 0.942, "ioi_gemma2": 2.721,
        "mcqa_qwen25": 0.093, "mcqa_gemma2": 0.565, "mcqa_llama3": 0.121,
        "arith_add_llama3": 0.092, "arith_sub_llama3": 0.032,
        "arc_easy_gemma2": 0.596, "arc_easy_llama3": 0.039,
        "arc_challenge_llama3": 0.698,
    },
    "IFR": {
        "ioi_gpt2": 0.415, "ioi_qwen25": 0.668, "ioi_gemma2": 0.749,
        "mcqa_qwen25": 0.485, "mcqa_gemma2": 0.673, "mcqa_llama3": 0.454,
        "arith_add_llama3": 0.285, "arith_sub_llama3": 0.247,
        "arc_easy_gemma2": 0.749, "arc_easy_llama3": 0.649,
    },
    "NAP_CF": {
        "ioi_gpt2": 0.721, "ioi_qwen25": 0.698, "ioi_gemma2": 0.668,
        "ioi_llama3": 0.739,
        "mcqa_qwen25": 0.718, "mcqa_gemma2": 0.628, "mcqa_llama3": 0.790,
        "arith_add_llama3": 0.748, "arith_sub_llama3": 0.707,
        "arc_easy_gemma2": 0.466, "arc_easy_llama3": 0.741,
        "arc_challenge_llama3": 0.744,
    },
    "NAP_IG_CF": {
        "ioi_gpt2": 0.248, "ioi_qwen25": 0.210, "ioi_gemma2": 0.664,
        "ioi_llama3": 0.574,
        "mcqa_qwen25": 0.473, "mcqa_gemma2": 1.119, "mcqa_llama3": 0.934,
        "arith_add_llama3": 0.528, "arith_sub_llama3": 0.601,
        "arc_easy_gemma2": 0.730, "arc_easy_llama3": 0.742,
        "arc_challenge_llama3": 0.743,
    },
    "UGS": {
        "ioi_gpt2": 0.035, "ioi_qwen25": 0.027,
        "mcqa_qwen25": 0.199,
    },
}
MIB_LEADERBOARD_CMD_CITE = Citation(
    paper=MIB.paper, venue=MIB.venue, arxiv_id=MIB.arxiv_id,
    location="HF space mib-bench/leaderboard (fixed x-axis percentages, differs from paper Table 2)",
)

MIB_LEADERBOARD_CPR: dict[str, dict[str, float]] = {
    "EAP_CF": {
        "ioi_gpt2": 1.201, "ioi_qwen25": 0.263, "ioi_gemma2": 1.275,
        "mcqa_qwen25": 0.752, "mcqa_gemma2": 1.301, "mcqa_llama3": 0.764,
        "arith_add_llama3": 0.537, "arith_sub_llama3": 0.675,
        "arc_easy_gemma2": 1.135, "arc_challenge_llama3": 0.256,
    },
    "EAP_IG_act_CF": {
        "ioi_gpt2": 0.322, "ioi_qwen25": 1.595, "ioi_gemma2": 0.713,
        "ioi_llama3": 1.053,
        "mcqa_qwen25": 0.776, "mcqa_gemma2": 1.334, "mcqa_llama3": 0.156,
        "arith_add_llama3": 0.477, "arith_sub_llama3": 0.968,
        "arc_easy_gemma2": 0.375, "arc_easy_llama3": 0.250,
        "arc_challenge_llama3": 0.263,
    },
    "EAP_IG_inp_CF": {
        "ioi_gpt2": 2.012, "ioi_qwen25": 1.940, "ioi_gemma2": 3.717,
        "mcqa_qwen25": 1.089, "mcqa_gemma2": 1.542, "mcqa_llama3": 1.087,
        "arith_add_llama3": 0.910, "arith_sub_llama3": 1.031,
        "arc_easy_gemma2": 1.586, "arc_easy_llama3": 1.038,
        "arc_challenge_llama3": 0.301,
    },
    "UGS": {
        "ioi_gpt2": 0.967, "ioi_qwen25": 0.980,
        "mcqa_qwen25": 1.184,
    },
}


# ---------------------------------------------------------------------------
# Known composition edges — GPT-2 Small
#
# Each entry: (write_layer, write_head, read_layer, read_head, composition_type)
# composition_type is "QK" or "OV" indicating which path the information
# flows through (QK = write output affects read's attention pattern,
# OV = write output is read by read's value/output path).
#
# These are the edges in Wang et al. Figure 2 and the implicit composition
# chains described in the text. The greater-than circuit edges are from
# Hanna et al. Section 3.2 (attention heads compose with MLPs, but
# head-to-head edges are less clearly specified).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CompositionEdge:
    write_layer: int
    write_head: int
    read_layer: int
    read_head: int
    composition_type: str
    write_role: str
    read_role: str
    task: str
    cite: str


KNOWN_COMPOSITION_EDGES: list[CompositionEdge] = [
    # ---- IOI: previous_token -> induction (QK) ----
    # Wang et al. Section 3.3: "previous token heads ... compose with
    # induction heads via QK composition"
    CompositionEdge(2, 2, 5, 5, "QK", "previous_token", "induction", "ioi",
                    "Wang et al. Section 3.3 + Figure 2"),
    CompositionEdge(2, 2, 6, 9, "QK", "previous_token", "induction", "ioi",
                    "Wang et al. Section 3.3 + Figure 2"),
    CompositionEdge(4, 11, 5, 5, "QK", "previous_token", "induction", "ioi",
                    "Wang et al. Section 3.3 + Figure 2"),
    CompositionEdge(4, 11, 6, 9, "QK", "previous_token", "induction", "ioi",
                    "Wang et al. Section 3.3 + Figure 2"),

    # ---- IOI: duplicate_token -> s_inhibition (QK) ----
    # Wang et al. Section 3.3: "duplicate token heads ... compose with
    # S-inhibition heads"
    CompositionEdge(0, 1, 7, 3, "QK", "duplicate_token", "s_inhibition", "ioi",
                    "Wang et al. Section 3.3 + Figure 2"),
    CompositionEdge(0, 1, 7, 9, "QK", "duplicate_token", "s_inhibition", "ioi",
                    "Wang et al. Section 3.3 + Figure 2"),
    CompositionEdge(0, 1, 8, 6, "QK", "duplicate_token", "s_inhibition", "ioi",
                    "Wang et al. Section 3.3 + Figure 2"),
    CompositionEdge(0, 1, 8, 10, "QK", "duplicate_token", "s_inhibition", "ioi",
                    "Wang et al. Section 3.3 + Figure 2"),
    CompositionEdge(3, 0, 7, 3, "QK", "duplicate_token", "s_inhibition", "ioi",
                    "Wang et al. Section 3.3 + Figure 2"),
    CompositionEdge(3, 0, 7, 9, "QK", "duplicate_token", "s_inhibition", "ioi",
                    "Wang et al. Section 3.3 + Figure 2"),
    CompositionEdge(3, 0, 8, 6, "QK", "duplicate_token", "s_inhibition", "ioi",
                    "Wang et al. Section 3.3 + Figure 2"),
    CompositionEdge(3, 0, 8, 10, "QK", "duplicate_token", "s_inhibition", "ioi",
                    "Wang et al. Section 3.3 + Figure 2"),

    # ---- IOI: induction -> s_inhibition (OV) ----
    # Wang et al. Section 3.2: induction heads write the S1 token to
    # positions where S-inhibition heads read it via OV path
    CompositionEdge(5, 5, 7, 3, "OV", "induction", "s_inhibition", "ioi",
                    "Wang et al. Section 3.2 + Figure 2"),
    CompositionEdge(5, 5, 7, 9, "OV", "induction", "s_inhibition", "ioi",
                    "Wang et al. Section 3.2 + Figure 2"),
    CompositionEdge(5, 5, 8, 6, "OV", "induction", "s_inhibition", "ioi",
                    "Wang et al. Section 3.2 + Figure 2"),
    CompositionEdge(5, 5, 8, 10, "OV", "induction", "s_inhibition", "ioi",
                    "Wang et al. Section 3.2 + Figure 2"),
    CompositionEdge(6, 9, 7, 3, "OV", "induction", "s_inhibition", "ioi",
                    "Wang et al. Section 3.2 + Figure 2"),
    CompositionEdge(6, 9, 7, 9, "OV", "induction", "s_inhibition", "ioi",
                    "Wang et al. Section 3.2 + Figure 2"),
    CompositionEdge(6, 9, 8, 6, "OV", "induction", "s_inhibition", "ioi",
                    "Wang et al. Section 3.2 + Figure 2"),
    CompositionEdge(6, 9, 8, 10, "OV", "induction", "s_inhibition", "ioi",
                    "Wang et al. Section 3.2 + Figure 2"),

    # ---- IOI: s_inhibition -> name_mover (QK) ----
    # Wang et al. Section 3.1: S-inhibition suppresses S1 attention in
    # name movers via QK path
    CompositionEdge(7, 3, 9, 6, "QK", "s_inhibition", "name_mover", "ioi",
                    "Wang et al. Section 3.1 + Figure 2"),
    CompositionEdge(7, 3, 9, 9, "QK", "s_inhibition", "name_mover", "ioi",
                    "Wang et al. Section 3.1 + Figure 2"),
    CompositionEdge(7, 3, 10, 0, "QK", "s_inhibition", "name_mover", "ioi",
                    "Wang et al. Section 3.1 + Figure 2"),
    CompositionEdge(7, 9, 9, 6, "QK", "s_inhibition", "name_mover", "ioi",
                    "Wang et al. Section 3.1 + Figure 2"),
    CompositionEdge(7, 9, 9, 9, "QK", "s_inhibition", "name_mover", "ioi",
                    "Wang et al. Section 3.1 + Figure 2"),
    CompositionEdge(7, 9, 10, 0, "QK", "s_inhibition", "name_mover", "ioi",
                    "Wang et al. Section 3.1 + Figure 2"),
    CompositionEdge(8, 6, 9, 6, "QK", "s_inhibition", "name_mover", "ioi",
                    "Wang et al. Section 3.1 + Figure 2"),
    CompositionEdge(8, 6, 9, 9, "QK", "s_inhibition", "name_mover", "ioi",
                    "Wang et al. Section 3.1 + Figure 2"),
    CompositionEdge(8, 6, 10, 0, "QK", "s_inhibition", "name_mover", "ioi",
                    "Wang et al. Section 3.1 + Figure 2"),
    CompositionEdge(8, 10, 9, 6, "QK", "s_inhibition", "name_mover", "ioi",
                    "Wang et al. Section 3.1 + Figure 2"),
    CompositionEdge(8, 10, 9, 9, "QK", "s_inhibition", "name_mover", "ioi",
                    "Wang et al. Section 3.1 + Figure 2"),
    CompositionEdge(8, 10, 10, 0, "QK", "s_inhibition", "name_mover", "ioi",
                    "Wang et al. Section 3.1 + Figure 2"),

    # ---- Induction: previous_token -> induction (K-composition) ----
    # Olsson et al. 2022: previous token heads shift token identity back
    # one position; induction heads match via K path
    *[CompositionEdge(wl, wh, rl, rh, "QK", "previous_token", "induction",
                      "induction", "Olsson et al. Section 2")
      for wl, wh in [(2, 2), (4, 11)]
      for rl, rh in [(5, 1), (5, 5), (6, 9), (7, 2), (7, 10)]],

    # ---- Greater-than: attention heads -> MLPs (residual stream) ----
    # Hanna et al. Section 3.2: 7 attention heads compose with MLPs 8-11.
    # read_head=-1 indicates MLP (no per-head index).
    # Only forward-causal edges (write_layer <= read_layer) included.
    *[CompositionEdge(wl, wh, rl, -1, "attn_to_mlp", "attention", "mlp",
                      "greater_than", "Hanna et al. Section 3.2")
      for wl, wh in [(5, 1), (5, 5), (6, 9), (7, 10), (8, 8), (8, 11), (9, 1)]
      for rl in [8, 9, 10, 11]
      if rl >= wl],

    # ---- SVA ----
    # TODO: Lazo et al. identify 12 important heads but do not publish
    # inter-head composition edges. Add when available.

    # ---- IOI: negative_name_movers (copy suppression) ----
    # Wang et al. Section 3: (10,7) and (11,10) suppress the IO token.
    # Self-contained — they read directly from the residual stream
    # without upstream composition edges.
]


def print_summary() -> None:
    """Print a compact comparison table to stdout."""
    print("\n=== Circuit Recovery Baselines (GPT-2 Small) ===\n")
    print(f"{'Task':<20} {'Metric':<18} {'Full':<8} {'Circuit':<10} {'Recovery':<10} {'Source'}")
    print("-" * 90)
    for task, info in CIRCUIT_RECOVERY.items():
        if task.startswith("greater_than_gen"):
            continue
        fm = info.get("full_model_score", "—")
        cs = info.get("circuit_score", "—")
        cite = str(info.get("_cite", ""))[:40]
        print(f"{task:<20} {str(info.get('metric', '')):<18} {str(fm):<8} {str(cs):<10} {info.get('recovery_pct', 0):.0f}%       {cite}")

    print("\n=== MIB Circuit CMD (lower=better), PMLR Table 2 ===\n")
    for model_task in ["gpt2_ioi", "qwen25_ioi", "gemma2_ioi", "llama31_ioi"]:
        scores = MIB_CIRCUIT_CMD.get(model_task, {})
        best_k = min((k for k, v in scores.items() if k != "random" and v is not None), key=lambda k: scores[k], default=None)
        best = scores.get(best_k) if best_k else None
        rand = scores.get("random")
        print(f"  {model_task:<25} best={best} ({best_k})  random={rand}")

    print("\n=== MIB Leaderboard CMD (different normalization!) ===\n")
    for model_task in ["ioi_gpt2", "ioi_qwen25", "ioi_gemma2", "ioi_llama3"]:
        paper_key = model_task.replace("ioi_", "").replace("qwen25", "qwen25") + "_ioi"
        paper_key = {"ioi_gpt2": "gpt2_ioi", "ioi_qwen25": "qwen25_ioi",
                     "ioi_gemma2": "gemma2_ioi", "ioi_llama3": "llama31_ioi"}[model_task]
        paper_best = min((v for k, v in MIB_CIRCUIT_CMD.get(paper_key, {}).items()
                         if k != "random" and v is not None), default=None)
        lb_scores = {m: d.get(model_task) for m, d in MIB_LEADERBOARD_CMD.items() if d.get(model_task) is not None}
        lb_best = min(lb_scores.values()) if lb_scores else None
        print(f"  {model_task:<20} paper={paper_best}  leaderboard={lb_best}")

    print(f"\n  Paper:  {MIB.paper}")
    print(f"  PMLR:   {MIB_PMLR}")
    print(f"  Raw:    reference/mib-leaderboard/eval-results-mib-subgraph/baselines/")


if __name__ == "__main__":
    print_summary()
