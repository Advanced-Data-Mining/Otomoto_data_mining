import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import ndarray
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix



def preprocessing(df, step: int = 20_000, min_price: int = 20_000, max_price: int = 300_000):
    """
    Function to preprocess the data
    :param df: Dataframe
    :param step: Step size for bins
    :param min_price: Minimum price for bins
    :param max_price: Maximum price for bins
    :return: df, bins
    """
    clean_df = df

    df = clean_df.drop(
        [
            "url",
            "color",
            "posted_date",
            "price_net_info",
            "location",
            "price",
            "country_of_origin",
        ],
        axis=1,
        errors="ignore",
    )

    df["capacity"] = (
        df["capacity"]
        .str.replace(" cm3", "", regex=False)
        .str.replace(" ", "")
        .astype(float)
    )

    df["power"] = (
        df["power"].str.replace(" ", "").str.replace("KM", "", regex=False).astype(float)
    )

    df["mileage"] = (
        df["mileage"].str.replace(" km", "", regex=False).str.replace(" ", "").astype(float)
    )

    df["price_pln"] = (
        df["price_pln"].str.replace(" ", "").str.replace(",", ".").astype(float)
    )

    df = df.dropna(
        subset=[
            "model",
            "condition",
            "fuel",
            "brand",
            "body_type",
            "accident_free",
            "year",
            "capacity",
            "power",
            "mileage",
            "seats",
            "description",
        ]
    )
    lin_bins = np.arange(min_price, max_price + step, step)  # edges: [min, min+step, ..., max]
    log_bins = np.logspace(
        np.log10(min_price),
        np.log10(max_price),
        num=len(lin_bins),
        dtype=float
    )
    log_bins = log_bins.astype(int)
    return df, lin_bins, log_bins


def custom_train_test_split(df, bins: ndarray, test_size: float = 0.2, random_state: int = 42, description_only = True):
    """
    Function to get train and test data
    :param random_state:
    :param test_size:
    :param df: Dataframe
    :param bins
    :return: X_train, X_test, y_train, y_test
    """
    df['price_bin'] = pd.cut(df['price_pln'], bins=bins, right=False, labels=False)

    print("Number of classes:", df['price_bin'].nunique())
    # delete data where price_bin is null
    df.dropna(subset=['price_bin'], inplace=True)

    # If too few rows remain, consider reducing min_samples_per_class or increasing step.

    # --- 4) train/test split (stratified) ---
    if description_only:
        X = df['description'].values
    else:
        X = df.drop(columns=["price_pln", "price_bin"])
    
    y = df['price_bin'].values
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )


def plot_hist(df, bins: ndarray, title: str, xlabel: str, ylabel: str):
    prices = df["price_pln"].values

    ranges = pd.cut(prices, bins=bins)
    counts = ranges.value_counts().sort_index()
    print("Min number of samples in bin: ", counts.min())
    print("Max number of samples in bin: ", counts.max())
    plt.figure(figsize=(14, 6))
    counts.plot(kind="bar")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_cm(cm, title: str = "Confusion Matrix", xlabel: str = "Predicted Label", ylabel: str = "True Label"):
    # --- PLOT ---
    plt.figure(figsize=(8, 8))
    im = plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # ticks
    plt.xticks(np.arange(cm.shape[1]))
    plt.yticks(np.arange(cm.shape[0]))
    
    maxv = cm.max()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]
            color = "black" if v > maxv * 0.40 else "white"
            plt.text(j, i, f"{v}",
                     ha="center", va="center",
                     color=color,
                     fontsize=9)

    plt.tight_layout()
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.grid(False)

    # show plot
    plt.show()

def generate_cm(y_test,y_pred):
    """
    Function to generate confusion matrix
    :param y_test:
    :param y_pred:
    :return: cm, cm_normalized
    """
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    return cm, cm_normalized.round(decimals=2)
