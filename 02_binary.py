"""
Experiment 2 – Binary classification: search vs. memorization.

The tuned logistic regression is saved and reused in Experiment 3.

Requires:  python 00_preprocess.py --mode search_vs_memorization
Usage:     python 02_binary.py
"""

import argparse
import pickle
from pathlib import Path

from src.config import FIGURES_DIR, MODE_SEARCH_VS_MEMO, MODELS_DIR, PROCESSED_DIR
from src.modeling import evaluate, get_models, load_features, plot_shap_beeswarms, tune

CLASSES = ["search", "memorization"]
FIGURE_DIR = FIGURES_DIR / "exp2_binary"
MODEL_PATH = MODELS_DIR / "lr_model_binary.pkl"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path,
                        default=PROCESSED_DIR / MODE_SEARCH_VS_MEMO,
                        help="Folder with the preprocessed CSV files.")
    return parser.parse_args()


def load_binary_split(data_dir, split_name):
    """Load a split and keep only search and memorization trials."""
    X, y = load_features(data_dir, split_name)
    mask = y.isin(CLASSES)
    return X.loc[mask], y.loc[mask]


def main():
    args = parse_args()
    X_train, y_train = load_binary_split(args.data_dir, "train")
    X_test, y_test = load_binary_split(args.data_dir, "test")

    for name, (estimator, param_grid) in get_models().items():
        model = tune(name, estimator, param_grid, X_train, y_train)
        evaluate(name, model, X_train, y_train, X_test, y_test, FIGURE_DIR)

        if name == "Logistic Regression":
            print("\n--- SHAP beeswarm (top-10 features) ---")
            plot_shap_beeswarms(model, X_test, FIGURE_DIR / "shap", n_top=10)

            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(MODEL_PATH, "wb") as f:
                pickle.dump(model, f)
            print(f"  Logistic regression saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
