from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src import config
from src.config import ModelType, ImputerStrategy
from src.preprocessing import build_imputer, build_scaler

_MODELS = {
    ModelType.LOGISTIC_REGRESSION: lambda: LogisticRegression(
        max_iter=1000,
        random_state=config.RANDOM_STATE,
    ),
    ModelType.KNN: lambda: KNeighborsClassifier(),
    ModelType.SVM: lambda: SVC(probability=True, random_state=config.RANDOM_STATE),
    ModelType.DECISION_TREE: lambda: DecisionTreeClassifier(
        random_state=config.RANDOM_STATE,
    ),
    ModelType.RANDOM_FOREST: lambda: RandomForestClassifier(
        random_state=config.RANDOM_STATE,
    ),
    ModelType.XGBOOST: lambda: XGBClassifier(
        random_state=config.RANDOM_STATE,
        n_jobs=1,
        eval_metric="logloss",
    ),
}

_PARAM_GRIDS = {
    ModelType.LOGISTIC_REGRESSION: {
        "model__C": [0.001, 0.01, 0.1, 1, 10, 100],
        "model__class_weight": [None, "balanced"],
    },

    ModelType.KNN: {
        "model__n_neighbors": [3, 5, 7, 9, 11, 15, 21, 31],
        "model__weights": ["uniform", "distance"],
        "model__p": [1, 2],
    },

    ModelType.SVM: {
        "model__C": [0.01, 0.1, 1, 10, 100],
        "model__kernel": ["linear", "rbf"],
        "model__gamma": ["scale", 0.001, 0.01, 0.1, 1],
        "model__class_weight": [None, "balanced"],
    },

    ModelType.DECISION_TREE: {
        "model__max_depth": [3, 4, 5, 7, 10, None],
        "model__min_samples_split": [2, 5, 10, 20],
        "model__min_samples_leaf": [1, 2, 5, 10],
        "model__criterion": ["gini", "entropy"],
        "model__class_weight": [None, "balanced"],
    },

    ModelType.RANDOM_FOREST: {
        "model__n_estimators": [300],
        "model__max_depth": [5, 10, 20, None],
        "model__max_features": ["sqrt", "log2", None],
        "model__min_samples_leaf": [1, 2, 5],
        "model__class_weight": [None, "balanced"],
    },

    ModelType.XGBOOST: {
        "model__n_estimators": [100, 300, 500, 800],
        "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "model__max_depth": [2, 3, 4, 5, 6],
        "model__subsample": [0.6, 0.8, 1.0],
        "model__colsample_bytree": [0.6, 0.8, 1.0],
        "model__reg_lambda": [0.1, 1, 10],
        "model__scale_pos_weight": [1, 1.87],
    },
}

def build_model(model_type):
    return _MODELS[ModelType(model_type)]()

def get_param_grid(model_type):
    return _PARAM_GRIDS[ModelType(model_type)]

def build_pipeline(model_type, imputer_strategy=ImputerStrategy.MEDIAN):
    model_type = ModelType(model_type)
    scaler = config.SCALER_BY_MODEL[model_type]
    return Pipeline([
        ("imputer", build_imputer(imputer_strategy)),
        ("scaler", build_scaler(scaler)),
        ("model", build_model(model_type)),
    ])