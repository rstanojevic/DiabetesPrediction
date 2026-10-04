from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

from src import config


def build_fold_matrix(df, best_imputers, metric=None):
    """Tabela: redovi su foldovi, kolone modeli, vrednosti metrika."""
    metric = metric or config.REFIT_METRIC
    mask = [best_imputers[m] == i for m, i in zip(df["model"], df["imputer"])]
    subset = df[mask]
    return subset.pivot_table(
        index=["repeat", "fold"], columns="model", values=metric
    )

def friedman(matrix):
    """Fridmanov test nad svim modelima, vraca (statistika, p)."""
    return friedmanchisquare(*[matrix[c] for c in matrix.columns])

def average_ranks(matrix):
    """Prosecan rang svakog modela (1 je najbolji)."""
    return matrix.rank(axis=1, ascending=False).mean().sort_values()

def holm_correction(pvalues):
    """Holmova korekcija za visestruko testiranje."""
    pvalues = np.asarray(pvalues)
    order = np.argsort(pvalues)
    m = len(pvalues)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * pvalues[idx])
        running_max = max(running_max, value)
        adjusted[idx] = running_max
    return adjusted

def pairwise_wilcoxon(matrix):
    """Wilcoxon test za svaki par modela, sa Holmovom korekcijom."""
    models = list(matrix.columns)
    rows, raw = [], []

    for a, b in combinations(models, 2):
        stat, p = wilcoxon(matrix[a], matrix[b])
        rows.append({
            "model_a": a,
            "model_b": b,
            "mean_a": matrix[a].mean(),
            "mean_b": matrix[b].mean(),
            "razlika": matrix[a].mean() - matrix[b].mean(),
            "p": p,
        })
        raw.append(p)

    out = pd.DataFrame(rows)
    out["p_holm"] = holm_correction(raw)
    out["znacajno"] = out["p_holm"] < 0.05
    return out.sort_values("p_holm").reset_index(drop=True)