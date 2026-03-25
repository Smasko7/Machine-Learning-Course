"""
preprocessing.py
----------------
Reusable preprocessing utilities extracted from the ML course exercises.

Covers:
- Feature scaling          (Exercises 1, 2, 4, 10)
- Missing value imputation (Exercises 2, 4)
- Text cleaning pipeline   (Exercise 7)
- Image loading & resizing (Exercises 5, 8)
"""

import re
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# ---------------------------------------------------------------------------
# Feature scaling
# ---------------------------------------------------------------------------

def scale_features(X_train, X_test, scaler: str = "minmax"):
    """Fit a scaler on X_train and transform both X_train and X_test.

    Parameters
    ----------
    X_train : array-like of shape (n_train, n_features)
    X_test  : array-like of shape (n_test, n_features)
    scaler  : "minmax" (default) or "standard"

    Returns
    -------
    X_train_scaled, X_test_scaled : np.ndarray
    fitted_scaler                 : the fitted scaler object
    """
    if scaler == "minmax":
        sc = MinMaxScaler()
    elif scaler == "standard":
        sc = StandardScaler()
    else:
        raise ValueError(f"Unknown scaler '{scaler}'. Choose 'minmax' or 'standard'.")

    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)
    return X_train_scaled, X_test_scaled, sc


# ---------------------------------------------------------------------------
# Missing value imputation
# ---------------------------------------------------------------------------

def impute_missing(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    """Fill missing values column-wise.

    Parameters
    ----------
    df       : pandas DataFrame
    strategy : "mean" (default), "median", or "mode"

    Returns
    -------
    df_filled : pandas DataFrame (copy)
    """
    df = df.copy()
    for col in df.columns:
        if df[col].isnull().any():
            if strategy == "mean":
                df[col] = df[col].fillna(df[col].mean())
            elif strategy == "median":
                df[col] = df[col].fillna(df[col].median())
            elif strategy == "mode":
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                raise ValueError(f"Unknown strategy '{strategy}'.")
    return df


# ---------------------------------------------------------------------------
# Text cleaning (Exercise 7 — Disaster Tweet Classification)
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Apply the full text-cleaning pipeline used in Exercise 7.

    Steps
    -----
    1. Remove non-ASCII / unicode characters
    2. Replace URLs with the token "url"
    3. Replace @mentions with the token "atUser"
    4. Remove # from hashtags (keep the word)
    5. Replace repeated punctuation (!!!, ???) with tokens
    6. Remove standalone numbers
    7. Strip extra whitespace

    Parameters
    ----------
    text : raw tweet string

    Returns
    -------
    cleaned : str
    """
    # Remove non-ASCII characters (e.g. emojis)
    text = text.encode("ascii", "ignore").decode("ascii")
    # Replace URLs
    text = re.sub(r"http\S+|www\.\S+", "url", text, flags=re.IGNORECASE)
    # Replace @mentions
    text = re.sub(r"@\w+", "atUser", text)
    # Remove # symbol from hashtags
    text = re.sub(r"#(\w+)", r"\1", text)
    # Replace repeated punctuation with tokens
    text = re.sub(r"!{2,}", " multiExclamation ", text)
    text = re.sub(r"\?{2,}", " multiQuestion ", text)
    text = re.sub(r"\.{2,}", " multiStop ", text)
    # Remove standalone numbers
    text = re.sub(r"\b\d+\b", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def clean_text_series(series: pd.Series) -> pd.Series:
    """Apply clean_text to every element of a pandas Series."""
    return series.astype(str).apply(clean_text)


# ---------------------------------------------------------------------------
# Image loading (Exercises 5, 8)
# ---------------------------------------------------------------------------

def load_images_from_folder(folder: str, img_size: tuple = (64, 64),
                             as_gray: bool = False) -> tuple:
    """Load all images from a folder tree and return arrays of pixels + labels.

    Expects the folder to contain one sub-directory per class:
        folder/
            class_a/img1.jpg
            class_a/img2.jpg
            class_b/img1.jpg

    Parameters
    ----------
    folder   : path to root folder
    img_size : (width, height) to resize images to
    as_gray  : if True, convert to grayscale

    Returns
    -------
    X : np.ndarray of shape (n_samples, height, width[, channels])
    y : list of string labels (class names)
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow is required: pip install Pillow")

    X, y = [], []
    for label in sorted(os.listdir(folder)):
        class_dir = os.path.join(folder, label)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            fpath = os.path.join(class_dir, fname)
            try:
                img = Image.open(fpath)
                img = img.resize(img_size)
                if as_gray:
                    img = img.convert("L")
                else:
                    img = img.convert("RGB")
                X.append(np.array(img))
                y.append(label)
            except Exception:
                pass  # skip unreadable files

    return np.array(X), y


def normalize_images(X: np.ndarray) -> np.ndarray:
    """Scale pixel values from [0, 255] to [0, 1]."""
    return X.astype(np.float32) / 255.0
