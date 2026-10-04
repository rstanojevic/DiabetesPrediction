from src import config


def select_best_imputer(df):
    """Za svaki model vraca imputacionu strategiju sa najboljim CV rezultatom."""
    best = (
        df.groupby(["model", "imputer"])[config.REFIT_METRIC]
        .mean()
        .reset_index()
        .sort_values(config.REFIT_METRIC, ascending=False)
        .drop_duplicates("model")
    )
    return dict(zip(best["model"], best["imputer"]))