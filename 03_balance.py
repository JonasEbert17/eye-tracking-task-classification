"""
Experiment 3 – Zero-shot classification of balance trials.

The logistic regression trained on search vs. memorization (Experiment 2) is
applied to balance trials, which it has never seen:

  a) full balance trials
  b) balance trials split into first and second half
  c) differential SHAP analysis: which features drive the change between halves?

Requires:  python 02_binary.py
           python 00_preprocess.py --mode balance_classification
Usage:     python 03_balance.py
"""

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import FIGURES_DIR, MODE_BALANCE, MODELS_DIR, PROCESSED_DIR
from src.modeling import compute_linear_shap_values, load_features, save_figure

FIGURE_DIR = FIGURES_DIR / "exp3_balance"
PARTS = ["balance_part1", "balance_part2"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=PROCESSED_DIR / MODE_BALANCE,
                        help="Folder with the preprocessed CSV files.")
    parser.add_argument("--model", type=Path, default=MODELS_DIR / "lr_model_binary.pkl",
                        help="Logistic regression saved by 02_binary.py.")
    return parser.parse_args()


def print_header(title):
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


def print_logits_and_probabilities(model, logits, proba, indent=""):
    print(f"{indent}Logits  –  mean: {logits.mean():.3f}   SD: {logits.std():.3f}")
    for k, cls in enumerate(model.classes_):
        print(f"{indent}P({cls})  –  mean: {proba[:, k].mean():.3f}   "
              f"SD: {proba[:, k].std():.3f}")


# =============================================================================
# 3a) Full balance trials
# =============================================================================

def analyse_full_trials(model, X):
    print_header("3a) Full balance trials")

    predictions = model.predict(X)
    print("\nPredicted label counts:")
    print(pd.Series(predictions).value_counts().to_string())
    print()
    print_logits_and_probabilities(model, model.decision_function(X), model.predict_proba(X))


# =============================================================================
# 3b) Split balance trials
# =============================================================================

def analyse_split_trials(model, X, y):
    print_header("3b) Split balance trials (part1 vs. part2)")

    predictions = model.predict(X)
    logits = model.decision_function(X)
    proba = model.predict_proba(X)

    # Confusion matrix: rows = true part, columns = predicted class (%)
    cm_parts = (pd.crosstab(y, predictions, normalize="index") * 100).reindex(
        index=PARTS, columns=list(model.classes_), fill_value=0)
    print("\nConfusion matrix – balance parts (row-normalised %):")
    print(cm_parts.round(2))

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm_parts, annot=True, fmt=".1f", cmap="Blues", cbar=False,
                annot_kws={"size": 10}, ax=ax)
    for text in ax.texts:
        text.set_text(text.get_text() + "%")
    ax.set_xlabel("Predicted Class", fontsize=10)
    ax.set_ylabel("True Balance Part", fontsize=10)
    ax.set_title("Balance Trial Parts – Classification (%)", fontsize=11)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    save_figure(fig, FIGURE_DIR / "confusion_matrix_balance_parts.png")

    print("\nLogits & probabilities per part:")
    for part in PARTS:
        idx = (y == part).to_numpy()
        print(f"\n  {part}:")
        print_logits_and_probabilities(model, logits[idx], proba[idx], indent="    ")


# =============================================================================
# 3c) Differential SHAP analysis
# =============================================================================

def analyse_shap_shift(model, X, y, n_top=10, n_labels=8):
    print_header("3c) Differential SHAP analysis (part2 − part1)")

    # Binary model: SHAP values refer to the positive class ('search')
    shap_values = compute_linear_shap_values(model, X)[-1]
    shap_df = pd.DataFrame(shap_values, columns=X.columns, index=X.index)

    is_part1 = (y == "balance_part1").to_numpy()
    is_part2 = (y == "balance_part2").to_numpy()

    delta_feature = X[is_part2].mean() - X[is_part1].mean()
    delta_shap = shap_df[is_part2].mean() - shap_df[is_part1].mean()

    shift = pd.DataFrame({
        "delta_feature": delta_feature,
        "delta_shap": delta_shap,
        "abs_delta_shap": delta_shap.abs(),
    }).sort_values("abs_delta_shap", ascending=False)

    print(f"\nTop-{n_top} features by |Δ SHAP| (part2 − part1):")
    print(shift.head(n_top).round(4).to_string())

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(shift["delta_feature"], shift["delta_shap"], alpha=0.7)
    ax.axhline(0, linestyle="--", linewidth=0.8, color="grey")
    ax.axvline(0, linestyle="--", linewidth=0.8, color="grey")
    for feature in shift.head(n_labels).index:
        ax.text(shift.loc[feature, "delta_feature"], shift.loc[feature, "delta_shap"],
                feature, fontsize=8)
    ax.set_xlabel("Δ Feature value (part2 − part1)")
    ax.set_ylabel("Δ SHAP impact (part2 − part1)")
    ax.set_title("Feature change vs. SHAP impact shift\n"
                 f"(positive SHAP = more '{model.classes_[1]}'-like)")
    fig.tight_layout()
    save_figure(fig, FIGURE_DIR / "shap_shift_part2_vs_part1.png")


def main():
    args = parse_args()

    with open(args.model, "rb") as f:
        model = pickle.load(f)

    X_full, _ = load_features(args.data_dir, "test_full")
    X_split, y_split = load_features(args.data_dir, "test_split")

    analyse_full_trials(model, X_full)
    analyse_split_trials(model, X_split, y_split)
    analyse_shap_shift(model, X_split, y_split)


if __name__ == "__main__":
    main()
