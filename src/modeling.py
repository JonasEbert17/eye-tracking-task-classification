"""
Shared helpers for the classification experiments: data loading, model
definitions with hyperparameter grids, evaluation and SHAP plots.
"""

import matplotlib
matplotlib.use("Agg")  # save figures to disk instead of opening windows

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier


# =============================================================================
# Data loading & saving figures
# =============================================================================

def load_features(data_dir, split_name):
    """Load X_<split_name>.csv and y_<split_name>.csv from `data_dir`."""
    X = pd.read_csv(data_dir / f"X_{split_name}.csv")
    y = pd.read_csv(data_dir / f"y_{split_name}.csv").squeeze("columns")
    return X, y


def save_figure(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {path}")


def slugify(name):
    return name.lower().replace(" ", "_")


# =============================================================================
# Models and hyperparameter grids
# =============================================================================

def get_models():
    """Return {model name: (estimator, parameter grid)} for all classifiers."""
    return {
        "Logistic Regression": (
            LogisticRegression(random_state=42, max_iter=1000),
            [
                {"penalty": ["l2"], "solver": ["lbfgs", "liblinear", "sag"],
                 "C": [0.001, 0.01, 0.1, 1, 10], "tol": [1e-4, 1e-3]},
                {"penalty": ["l1"], "solver": ["liblinear"],
                 "C": [0.001, 0.01, 0.1, 1, 10], "tol": [1e-4, 1e-3]},
                {"penalty": ["elasticnet"], "solver": ["saga"],
                 "C": [0.001, 0.01, 0.1, 1, 10], "l1_ratio": [0.25, 0.5, 0.75],
                 "tol": [1e-4, 1e-3]},
            ],
        ),
        "AdaBoost": (
            AdaBoostClassifier(estimator=DecisionTreeClassifier(random_state=0),
                               random_state=0),
            {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.5, 1.0],
                "estimator__max_depth": [1, 2, 3],
                "estimator__min_samples_split": [2, 5, 10],
                "estimator__min_samples_leaf": [1, 2, 5],
            },
        ),
        "HistGradientBoosting": (
            HistGradientBoostingClassifier(random_state=0),
            {
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5, 10, None],
                "min_samples_leaf": [20, 50, 100],
                "max_iter": [100, 200, 300],
            },
        ),
        "LightGBM": (
            LGBMClassifier(random_state=0, verbose=-1),
            {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.05, 0.1],
                "num_leaves": [31, 50, 100],
                "max_depth": [-1, 5, 10],
                "subsample": [0.7, 0.9, 1.0],
            },
        ),
        "MLP": (
            MLPClassifier(validation_fraction=0.1, early_stopping=True, tol=1e-4,
                          momentum=0.9, beta_1=0.9, beta_2=0.999,
                          random_state=42, verbose=False),
            {
                "solver": ["sgd", "adam"],
                "activation": ["logistic", "relu"],
                "hidden_layer_sizes": [(10,), (50,), (100,)],
                "learning_rate_init": [0.01, 0.001, 0.003],
                "alpha": [0.0001, 0.001],
                "max_iter": [200, 400],
            },
        ),
    }


def tune(name, estimator, param_grid, X_train, y_train):
    """5-fold grid search optimising balanced accuracy; returns best model."""
    search = GridSearchCV(estimator, param_grid, cv=5,
                          scoring="balanced_accuracy", n_jobs=-1, verbose=1)
    search.fit(X_train, y_train)
    print(f"  Best params ({name}): {search.best_params_}")
    print(f"  Best CV balanced accuracy: {search.best_score_:.3f}")
    return search.best_estimator_


# =============================================================================
# Evaluation
# =============================================================================

def plot_confusion_matrix(y_true, y_pred, title):
    """Row-normalised confusion matrix (%) as heatmap."""
    labels = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    size = max(5, len(labels))
    fig, ax = plt.subplots(figsize=(size, size - 1))
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    ax.set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)),
           xticklabels=labels, yticklabels=labels,
           xlabel="Predicted label", ylabel="True label", title=title)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = cm_pct[i, j]
            ax.text(j, i, f"{value:.1f}%", ha="center", va="center",
                    color="white" if value > 50 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, label="%")
    fig.tight_layout()
    return fig


def evaluate(name, model, X_train, y_train, X_test, y_test, figure_dir):
    """Print performance metrics and save the confusion matrix."""
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print(f"\n{'=' * 60}\n  {name}\n{'=' * 60}")
    print(f"  Train accuracy : {accuracy_score(y_train, y_train_pred):.3f}")
    print(f"  Test accuracy  : {accuracy_score(y_test, y_test_pred):.3f}")
    print(f"  Weighted F1    : {f1_score(y_test, y_test_pred, average='weighted'):.3f}")
    print("\n  Classification report:")
    print(classification_report(y_test, y_test_pred, digits=3))

    fig = plot_confusion_matrix(y_test, y_test_pred,
                                title=f"{name} – Confusion Matrix (%)")
    save_figure(fig, figure_dir / f"confusion_matrix_{slugify(name)}.png")
    return y_test_pred


# =============================================================================
# SHAP (logistic regression only)
# =============================================================================

def compute_linear_shap_values(model, X):
    """
    SHAP values of a linear model as a list with one (n_samples, n_features)
    array per class. For binary models the list has a single entry, which
    refers to the positive class `model.classes_[1]`.
    """
    X = np.asarray(X)
    n_features = X.shape[1]

    explainer = shap.LinearExplainer(model, X, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X)

    # The output format differs between shap versions and binary/multiclass
    if isinstance(shap_values, list):
        return [sv[:, :n_features] for sv in shap_values]
    if shap_values.ndim == 3:
        return [shap_values[:, :n_features, k] for k in range(shap_values.shape[2])]
    return [shap_values[:, :n_features]]


def plot_shap_beeswarms(model, X_df, figure_dir, n_top=10):
    """Save one SHAP beeswarm plot (top-n features) per class."""
    X = X_df.to_numpy()
    feature_names = np.array(X_df.columns)

    shap_per_class = compute_linear_shap_values(model, X)
    class_labels = (model.classes_ if len(shap_per_class) > 1
                    else [model.classes_[1]])

    # Common x-axis limits for comparability across classes
    max_abs = max(np.abs(sv).max() for sv in shap_per_class)

    for sv, label in zip(shap_per_class, class_labels):
        top_idx = np.argsort(np.mean(np.abs(sv), axis=0))[-n_top:][::-1]

        plt.figure()
        shap.summary_plot(sv[:, top_idx], X[:, top_idx],
                          feature_names=feature_names[top_idx], show=False)
        plt.xlim(-max_abs, max_abs)
        plt.title(f"SHAP Beeswarm – class: {label}")
        plt.tight_layout()
        save_figure(plt.gcf(), figure_dir / f"shap_beeswarm_{slugify(str(label))}.png")
