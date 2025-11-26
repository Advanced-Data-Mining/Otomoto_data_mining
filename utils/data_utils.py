import math

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def preprocessing(df, step: int = 50_000):
    """
    Function to preprocess the data
    :param step: Step size for bins
    :param df: Dataframe
    :return: df, bins
    """
    # --- 1) ensure numeric price and clean text ---
    df['price_pln'] = pd.to_numeric(
        df['price_pln'].astype(str).str.replace(r'[^0-9\.-]', '', regex=True),
        errors='coerce'
    )

    df['description'] = df['description'].astype(str).fillna('').str.strip()

    # drop rows missing price or description
    df = df[df['price_pln'].notna() & (df['description'] != '')].copy()

    # --- 2) make 50_000-PLN bins (auto range) ---
    min_price = int(math.floor(df['price_pln'].min() / step) * step)
    max_price = int(math.ceil(df['price_pln'].max() / step) * step)
    bins = np.arange(min_price, max_price + step, step)  # edges: [min, min+step, ..., max]
    print(f"Bins from {min_price} to {max_price} step {step} -> {len(bins) - 1} classes")

    # assign bin labels (0..n_bins-1)
    df['price_bin'] = pd.cut(df['price_pln'], bins=bins, right=False, labels=False)

    # drop out-of-range just in case
    df = df[df['price_bin'].notna()].copy()
    df['price_bin'] = df['price_bin'].astype(int)
    return df, bins


def custom_train_test_split(df, min_samples: int = 10):
    """
    Function to get train and test data
    :param df: Dataframe
    :param min_samples: minimum number of samples needed for training
    :return: X_train, X_test, y_train, y_test
    """
    counts = df['price_bin'].value_counts()
    valid_bins = counts[counts >= min_samples].index
    df = df[df['price_bin'].isin(valid_bins)].copy()
    print("Rows after filtering rare bins:", len(df))
    print("Number of classes after filtering:", df['price_bin'].nunique())

    # If too few rows remain, consider reducing min_samples_per_class or increasing step.

    # --- 4) train/test split (stratified) ---
    X = df['description'].values
    y = df['price_bin'].values
    return train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
