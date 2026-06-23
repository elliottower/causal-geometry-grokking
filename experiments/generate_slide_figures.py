"""Generate just the keeper figures for the slides presentation.

Outputs to slides/figures/generated/

Usage:
    uv run --with matplotlib python experiments/generate_slide_figures.py
"""

from pathlib import Path
import sys

# Override OUTPUT_DIR before importing
import generate_figures

OUT = Path(__file__).parent.parent / "slides" / "figures" / "generated"
OUT.mkdir(parents=True, exist_ok=True)
generate_figures.OUTPUT_DIR = OUT

KEEPERS = [
    generate_figures.fig7_loss_vs_equiv,        # loss vs equiv (grokked/not scatter)
    generate_figures.fig12_loss_vs_equiv_paper,  # paper-quality loss vs equiv
    generate_figures.fig13_polynomial_degree_ladder,  # polynomial degree ladder
    generate_figures.fig14_k_sweep,             # k-sweep profiles
    generate_figures.fig15d_sorted_dot,         # sorted equivariance dot plot
    generate_figures.fig20_real_circles,        # real DAS projections from JSONL
    generate_figures.fig21_three_classes,       # three-class partition
    generate_figures.fig22_stochastic_trajectories,  # stochastic trajectories
    generate_figures.fig23b_k_sweep_equiv_colored,   # k-sweep colored by equiv
    generate_figures.fig24_three_class_trajectories,  # three-class training curves
]

def main():
    print(f"Output directory: {OUT}")
    for fn in KEEPERS:
        try:
            fn()
        except Exception as e:
            print(f"  SKIPPED {fn.__name__}: {e}")
    print(f"\nDone — {len(KEEPERS)} figures attempted, saved to {OUT}")

if __name__ == "__main__":
    main()
