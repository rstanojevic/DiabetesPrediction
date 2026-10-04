from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from src import config
from src.config import ImputerStrategy, ScalerType

_IMPUTERS = {
    ImputerStrategy.MEAN: lambda: SimpleImputer(strategy="mean"),
    ImputerStrategy.MEDIAN: lambda: SimpleImputer(strategy="median"),
    ImputerStrategy.KNN: lambda: KNNImputer(
        n_neighbors=config.KNN_IMPUTER_NEIGHBORS
    ),
    ImputerStrategy.ITERATIVE: lambda: IterativeImputer(
        max_iter=config.ITERATIVE_IMPUTER_MAX_ITER,
        random_state=config.RANDOM_STATE,
    ),
}

_SCALERS = {
    ScalerType.STANDARD: lambda: StandardScaler(),
    ScalerType.ROBUST: lambda: RobustScaler(),
    ScalerType.MINMAX: lambda: MinMaxScaler(),
    ScalerType.NONE: lambda: "passthrough",
}


def build_imputer(strategy=ImputerStrategy.MEDIAN):
    return _IMPUTERS[ImputerStrategy(strategy)]()


def build_scaler(scaler=ScalerType.STANDARD):
    return _SCALERS[ScalerType(scaler)]()


def build_preprocessor(
    imputer_strategy=ImputerStrategy.MEDIAN,
    scaler=ScalerType.STANDARD,
):
    """Sastavlja pipeline: imputacija pa skaliranje.

    Pipeline se uci iskljucivo na trening podacima, sto sprecava
    curenje informacija iz test skupa.
    """
    return Pipeline([
        ("imputer", build_imputer(imputer_strategy)),
        ("scaler", build_scaler(scaler)),
    ])