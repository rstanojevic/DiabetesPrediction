from pathlib import Path

from enum import Enum

from sklearn.metrics import make_scorer, f1_score, precision_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = REPORTS_DIR / "logs"

DATASET_PATH = DATA_RAW_DIR / "diabetes.csv"

TARGET = "Outcome"
FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 10
CV_REPEATS = 5

FIGURE_DPI = 300
FIGURE_FORMAT = "pdf"
FIGSIZE_SINGLE = (7, 5)
FIGSIZE_GRID = (14, 10)
FIGSIZE_SQUARE = (9, 8)
FONT_SIZE = 11

KNN_IMPUTER_NEIGHBORS = 5
ITERATIVE_IMPUTER_MAX_ITER = 10

class ImputerStrategy(str, Enum):
    """Nacin dopune nedostajucih vrednosti."""
    MEAN = "mean"
    MEDIAN = "median"
    KNN = "knn"
    ITERATIVE = "iterative"


class ScalerType(str, Enum):
    """Nacin skaliranja atributa."""
    STANDARD = "standard"
    ROBUST = "robust"
    MINMAX = "minmax"
    NONE = "none"


SCALER_BY_MODEL = {
    "logistic_regression": ScalerType.STANDARD,
    "knn": ScalerType.STANDARD,
    "svm": ScalerType.STANDARD,
    "decision_tree": ScalerType.NONE,
    "random_forest": ScalerType.NONE,
    "xgboost": ScalerType.NONE,
}

class ModelType(str, Enum):
    LOGISTIC_REGRESSION = "logistic_regression"
    KNN = "knn"
    SVM = "svm"
    DECISION_TREE = "decision_tree"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"

SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": "recall",
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
}

REFIT_METRIC = "roc_auc"
RANDOMIZED_SEARCH_MODELS = [ModelType.XGBOOST, ModelType.DECISION_TREE]
N_ITER_RANDOM_SEARCH = 100

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "roc_auc": "ROC-AUC",
    "average_precision": "PR-AUC",
}

# Referentne vrednosti trivijalnog modela, izvedene iz raspodele klasa
# (500 negativnih / 268 pozitivnih; udeo pozitivnih = 0.349)
METRIC_BASELINES = {
    "accuracy": 0.651,            # uvek predvidja vecinsku klasu
    "precision": 0.349,           # nasumicno pogadjanje
    "roc_auc": 0.5,               # nasumicno rangiranje
    "average_precision": 0.349,   # udeo pozitivnih
}