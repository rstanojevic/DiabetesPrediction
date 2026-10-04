"""SHAP analiza najboljeg modela na test skupu."""
import shap
from matplotlib import pyplot as plt

from src.data_loader import split_data
from src.explainability import load_pipeline, compute_shap_values
from src.plotting import setup_plotting, save_figure

MODEL = "xgboost"
IMPUTER = "iterative"

def main():
    setup_plotting()

    _, X_test, _, y_test = split_data()
    pipe = load_pipeline(MODEL, IMPUTER)
    shap_values, X_transformed = compute_shap_values(pipe, X_test)

    shap.plots.bar(shap_values, show=False)
    save_figure(plt.gcf(), "fig_09_shap_importance")

    shap.plots.beeswarm(shap_values, show=False)
    save_figure(plt.gcf(), "fig_10_shap_beeswarm")

    y_proba = pipe.predict_proba(X_test)[:,1]
    idx = int(y_proba.argmax())
    shap.plots.waterfall(shap_values[idx], show=False)
    save_figure(plt.gcf(), "fig_11_shap_waterfall")

    print(f"Pacijentkinja sa najvecom verovatnocom: p = {y_proba[idx]:.3f}, "
          f"stvarna klasa = {y_test.iloc[idx]}")

if __name__ == "__main__":
    main()