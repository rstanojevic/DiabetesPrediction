import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src import config
import seaborn as sns

from src.evaluation import select_best_imputer
from src.plotting import save_figure, setup_plotting
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    precision_score, recall_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve,
)
import joblib
from src.config import ModelType
from src.data_loader import split_data
from src.statistics import build_fold_matrix, friedman, average_ranks, pairwise_wilcoxon


def summarize_cv_results(df):
    metrics = list(config.SCORING)

    by_model = df.groupby("model")[metrics].agg(["mean", "std"])
    by_model.round(4).to_csv(config.TABLES_DIR / "summary_by_model.csv")

    by_both = df.groupby(["model", "imputer"])[metrics].agg(["mean", "std"])
    by_both.round(4).to_csv(config.TABLES_DIR / "summary_by_model_imputer.csv")

    print(by_model[config.REFIT_METRIC].round(4))

def plot_metric_boxplots(df):
    """Boxplot svake metrike po modelu, preko svih foldova i imputacija."""
    metrics = list(config.SCORING)

    order = (
        df.groupby("model")[config.REFIT_METRIC]
        .median()
        .sort_values(ascending=False)
        .index
    )

    fig, axes = plt.subplots(2, 3, figsize=config.FIGSIZE_GRID)
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        sns.boxplot(data=df, x="model", y=metric, order=order,
                    hue="model", hue_order=order, legend=False, ax=ax)
        ax.set_title(config.METRIC_LABELS.get(metric, metric))
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=45, ha="right")
        baseline = config.METRIC_BASELINES.get(metric)
        if baseline is not None:
            ax.axhline(baseline, ls="--", lw=1, color="gray")
            ax.text(
                0.99, baseline, f" {baseline:.3f}",
                transform=ax.get_yaxis_transform(),
                va="bottom", ha="right", fontsize=config.FONT_SIZE - 2, color="gray",
            )

    fig.suptitle(
        "Raspodela metrika po modelu (50 foldova x 4 imputacione strategije)",
        fontsize=config.FONT_SIZE + 4,
    )
    fig.tight_layout()
    save_figure(fig, "fig_06_metric_boxplots")

def evaluate_on_test(df, X_test, y_test):
    """Konacna ocena na zakljucanom test skupu."""
    best_imputers = select_best_imputer(df)

    rows = []
    for model_type in ModelType:
        imputer = best_imputers[model_type.value]
        pipe = joblib.load(
            config.MODELS_DIR / f"{model_type.value}_{imputer}.pkl"
        )

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        rows.append({
            "model": model_type.value,
            "imputer": imputer,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "average_precision": average_precision_score(y_test, y_proba),
        })

    out = (
        pd.DataFrame(rows)
        .sort_values(config.REFIT_METRIC, ascending=False)
        .reset_index(drop=True)
    )
    out.round(4).to_csv(config.TABLES_DIR / "test_results.csv", index=False)
    print(out.round(4).to_string(index=False))
    return out

def plot_confusion_matrices(df, X_test, y_test):
    """Matrice konfuzije na test skupu, normalizovane po stvarnoj klasi."""
    best_imputers = select_best_imputer(df)
    labels = ["Zdrava", "Bolesna"]

    fig, axes = plt.subplots(2, 3, figsize=config.FIGSIZE_GRID)
    axes = axes.flatten()

    for i, model_type in enumerate(ModelType):
        ax = axes[i]
        imputer = best_imputers[model_type.value]
        pipe = joblib.load(
            config.MODELS_DIR / f"{model_type.value}_{imputer}.pkl"
        )
        y_pred = pipe.predict(X_test)

        cm_abs = confusion_matrix(y_test, y_pred)
        cm_norm = confusion_matrix(y_test, y_pred, normalize="true")

        annot = np.array([
            [f"{cm_norm[r, c]:.2f}\n({cm_abs[r, c]})" for c in range(2)]
            for r in range(2)
        ])

        sns.heatmap(cm_norm, annot=annot, fmt="", cmap="Blues",
                    vmin=0, vmax=1, cbar=False, square=True,
                    xticklabels=labels, yticklabels=labels, ax=ax,
        )

        ax.set_title(f"{model_type.value} ({imputer})")
        ax.set_xlabel("Predvidjeno")
        ax.set_ylabel("Stvarno")

    fig.suptitle("Matrice konfuzije na test skupu (n=154)",
                    fontsize=config.FONT_SIZE + 4
    )

    fig.tight_layout()
    save_figure(fig, "fig_07_confusion_matrices")

def plot_curves(df, X_test, y_test):
    """ROC i PR krive svih modela na test skupu."""
    best_imputers = select_best_imputer(df)
    positive_rate = y_test.mean()

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=config.FIGSIZE_GRID)

    for model_type in ModelType:
        imputer = best_imputers[model_type.value]
        pipe = joblib.load(
            config.MODELS_DIR / f"{model_type.value}_{imputer}.pkl"
        )
        y_proba = pipe.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax_roc.plot(fpr, tpr, label=f"{model_type.value} ({auc:.3f})")

        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        ax_pr.plot(recall, precision, label=f"{model_type.value} ({ap:.3f})")

    ax_roc.plot([0, 1], [0, 1], ls="--", lw=1, color="grey")
    ax_roc.set_xlabel("FPR (udeo laznih uzbuna)")
    ax_roc.set_ylabel("TPR (recall)")
    ax_roc.set_title("ROC krive")
    ax_roc.legend(loc="lower right")

    ax_pr.axhline(positive_rate, ls="--", lw=1, color="grey")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("PR krive")
    ax_pr.legend(loc="upper right")

    fig.suptitle(
        "ROC i PR krive na test skupu", size=config.FONT_SIZE + 4
    )

    fig.tight_layout()
    save_figure(fig, "fig_08_roc_pr_curves")

def run_statistical_tests(df):
    """Friedman nad svim modelima pa Wilcoxon po parovima"""
    best_imputers = select_best_imputer(df)
    matrix = build_fold_matrix(df, best_imputers)

    stat, p = friedman(matrix)
    print(f"\nFriedmanov test: chi2 = {stat:.3f}, p = {p:.2e}")

    ranks = average_ranks(matrix)
    print("\nProsecni rangovi (1 = najbolji):")
    print(ranks.round(3).to_string())
    ranks.round(3).to_csv(config.TABLES_DIR / "average_ranks.csv")

    pairs = pairwise_wilcoxon(matrix)
    print("\nWilcoxon po parovima:")
    print(pairs.round(4).to_string(index=False))
    pairs.round(6).to_csv(config.TABLES_DIR / "wilcoxon_pairwise.csv", index=False)

    znacajnih = pairs["znacajno"].sum()
    print(f"\nZnacajnih razlika: {znacajnih} od {len(pairs)} parova")

    return matrix, pairs

def main():
    setup_plotting()
    df = pd.read_csv(config.TABLES_DIR / "cv_results.csv")
    summarize_cv_results(df)
    plot_metric_boxplots(df)
    X_train, X_test, y_train, y_test = split_data()
    evaluate_on_test(df, X_test, y_test)
    plot_confusion_matrices(df, X_test, y_test)
    plot_curves(df, X_test, y_test)
    run_statistical_tests(df)

if __name__ == "__main__":
    main()