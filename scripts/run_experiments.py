import time
import warnings
import logging
from datetime import datetime

import joblib
import pandas as pd
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
)

from src import config
from src.config import ImputerStrategy, ModelType
from src.data_loader import split_data
from src.models import build_pipeline, get_param_grid

warnings.filterwarnings("ignore", category=FutureWarning)
config.LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(config.LOGS_DIR / "experiments.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

def build_cv():
    """Ponovljena stratifikovana k-fold kros-validacija."""
    return RepeatedStratifiedKFold(
        n_splits=config.CV_FOLDS,
        n_repeats=config.CV_REPEATS,
        random_state=config.RANDOM_STATE,
    )


def extract_fold_results(search, model_type, imputer_strategy):
    """Izvlaci rezultate po pojedinacnom foldu za najbolju kombinaciju.

    GridSearchCV u cv_results_ cuva ocenu svakog folda za svaku
    kombinaciju, pod kljucevima oblika split<i>_test_<metrika>.
    Uzimaju se samo oni koji odgovaraju best_index_.
    """
    results = search.cv_results_
    best = search.best_index_
    n_folds = config.CV_FOLDS * config.CV_REPEATS

    rows = []
    for fold in range(n_folds):
        row = {
            "model": model_type.value,
            "imputer": ImputerStrategy(imputer_strategy).value,
            "repeat": fold // config.CV_FOLDS,
            "fold": fold % config.CV_FOLDS,
        }
        for metric in config.SCORING:
            key = f"split{fold}_test_{metric}"
            row[metric] = results[key][best]
        rows.append(row)
    return rows

def run_single_experiment(model_type, imputer_strategy, X_train, y_train):
    pipeline = build_pipeline(model_type, imputer_strategy)
    param_grid = get_param_grid(model_type)
    if ModelType(model_type) in config.RANDOMIZED_SEARCH_MODELS:
        search = RandomizedSearchCV(
            estimator=pipeline,
            scoring=config.SCORING,
            refit=config.REFIT_METRIC,
            cv=build_cv(),
            n_jobs=-1,
            verbose=1,
            param_distributions=param_grid,
            n_iter=config.N_ITER_RANDOM_SEARCH,
            random_state=config.RANDOM_STATE,
        )
    else:
        search = GridSearchCV(
            estimator=pipeline,
            scoring=config.SCORING,
            refit=config.REFIT_METRIC,
            cv=build_cv(),
            n_jobs=-1,
            verbose=1,
            param_grid=param_grid,
        )
    search.fit(X_train, y_train)
    return search

def main():
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)

    logging.info(f"Pokretanje: {datetime.now():%Y-%m-%d %H:%M}\n")

    X_train, X_test, y_train, y_test = split_data()

    all_rows = []
    for model_type in ModelType:
        for imputer_strategy in ImputerStrategy:
            start = time.time()
            search = run_single_experiment(model_type, imputer_strategy, X_train, y_train)
            all_rows.extend(extract_fold_results(search, model_type, imputer_strategy))
            duration = time.time() - start
            logging.info(f"\n{'=' * 70}")
            logging.info(f"{model_type.value}  |  imputacija: {imputer_strategy.value}")
            logging.info(f"{'=' * 70}")
            logging.info(f"Najbolji {config.REFIT_METRIC}: {search.best_score_:.4f}")
            logging.info(f"Trajanje: {duration:.1f}s")
            logging.info("Hiperparametri:")
            for param, value in sorted(search.best_params_.items()):
                logging.info(f"   {param.replace('model__', ''):<24} {value}")
            model_path = config.MODELS_DIR / f"{model_type.value}_{imputer_strategy.value}.pkl"
            joblib.dump(search.best_estimator_, model_path)

    pd.DataFrame(all_rows).to_csv(
        config.TABLES_DIR / "cv_results.csv", index=False
    )

if __name__ == "__main__":
    main()