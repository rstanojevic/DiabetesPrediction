import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config

def load_raw():
    """Vraca skup podataka tacno onakav kakav je u CSV fajlu."""
    return pd.read_csv(config.DATASET_PATH)

def replace_zeros_with_nan(df):
    """Zamenjuje fizioloski nemoguce nule NaN vrednostima. Radi nad kopijom, tako da ulazni DataFrame ostaje netaknut."""
    df = df.copy()
    df[config.ZERO_AS_MISSING] = df[config.ZERO_AS_MISSING].replace(0, np.nan)
    return df

def split_features_target(df):
    """Razdvaja matricu atributa X i vektor oznaka y."""
    X = df[config.FEATURES]
    y = df[config.TARGET]
    return X, y

def split_data():
    """Deli podatke na trening i zakljucan test skup."""
    df = replace_zeros_with_nan(load_raw())
    X, y = split_features_target(df)
    return train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )