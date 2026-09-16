"""
Experiment 1 – Multiclass classification of all four viewing tasks
(free viewing, search, memorization, balance).

Requires:  python 00_preprocess.py --mode all
Usage:     python 01_multiclass.py
"""

import argparse
from pathlib import Path

from src.config import FIGURES_DIR, MODE_ALL, PROCESSED_DIR
from src.modeling import evaluate, get_models, load_features, plot_shap_beeswarms, tune

FIGURE_DIR = FIGURES_DIR / "exp1_multiclass"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=PROCESSED_DIR / MODE_ALL,
                        help="Folder with the preprocessed CSV files.")
    return parser.parse_args()


def main():
    args = parse_args()
    X_train, y_train = load_features(args.data_dir, "train")
    X_test, y_test = load_features(args.data_dir, "test")

    for name, (estimator, param_grid) in get_models().items():
        model = tune(name, estimator, param_grid, X_train, y_train)
        evaluate(name, model, X_train, y_train, X_test, y_test, FIGURE_DIR)

        if name == "Logistic Regression":
            print("\n--- SHAP beeswarm per class (top-10 features) ---")
            plot_shap_beeswarms(model, X_test, FIGURE_DIR / "shap", n_top=10)


if __name__ == "__main__":
    main()
