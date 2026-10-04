import joblib
import pandas as pd
import shap

from src import config


def load_pipeline(model_type, imputer_strategy):
    return joblib.load(config.MODELS_DIR / f"{model_type}_{imputer_strategy}.pkl")

def transform_features(pipe, X):
    """Primenjuje korake pretprocesiranja, bez modela.

    SHAP radi nad ulazom koji model zaista vidi, dakle nakon
    imputacije i skaliranja. Nazivi kolona se vracaju rucno,
    jer transformacija vraca numpy niz.
    """
    X_transformed = pipe[:-1].transform(X)
    return pd.DataFrame(X_transformed, columns=config.FEATURES)

def compute_shap_values(pipe, X):
    """Racuna SHAP vrednosti za model zasnovan na stablima"""
    model = pipe.named_steps["model"]
    X_transformed = transform_features(pipe, X)
    explainer = shap.TreeExplainer(model)
    return explainer(X_transformed), X_transformed