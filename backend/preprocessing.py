"""
Prétraitement partagé (entraînement ET API) : nettoyage, extraction de
caractéristiques et normalisation.
"""
import json
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

V_COLS = [f"V{i}" for i in range(1, 29)]
RAW_COLS = V_COLS + ["Time", "Amount"]
# Variables extraites en plus des 30 colonnes d'origine
EXTRA_COLS = ["hour", "is_night", "amount_log"]
FEATURES = RAW_COLS + EXTRA_COLS
# Colonnes mises à l'échelle (les V1-V28 sont déjà centrées/réduites par la PCA)
SCALED_COLS = ["Time", "Amount", "amount_log"]

MODELS_DIR = "models"
FEATURES_FILE = os.path.join(MODELS_DIR, "feature_columns.json")


def clean(df: pd.DataFrame, training: bool = True) -> pd.DataFrame:
    """
    Nettoyage.
    """
    out = df.copy()
    for c in RAW_COLS:
        if c not in out.columns:
            out[c] = 0.0
        out[c] = pd.to_numeric(out[c], errors="coerce")
    if training:
        out = out.drop_duplicates(subset=RAW_COLS)
        out = out.dropna(subset=RAW_COLS)
    else:
        out[RAW_COLS] = out[RAW_COLS].fillna(0.0)
    out["Amount"] = out["Amount"].clip(lower=0)
    out["Time"] = out["Time"].clip(lower=0)
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extraction de caractéristiques : heure, nuit, montant logarithmique."""
    X = df.copy()
    X["hour"] = ((X["Time"] // 3600) % 24).astype(float)
    X["is_night"] = (X["hour"] <= 5).astype(float)
    X["amount_log"] = np.log1p(X["Amount"])
    return X[FEATURES]


def fit_scaler(df: pd.DataFrame) -> StandardScaler:
    """Calcule le scaler sur les données après extraction des features."""
    X = add_features(clean(df, training=True))
    scaler = StandardScaler()
    scaler.fit(X[SCALED_COLS])
    return scaler


def normalize(X: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    X = X.copy()
    X[SCALED_COLS] = scaler.transform(X[SCALED_COLS])
    return X


def prepare(df: pd.DataFrame, scaler: StandardScaler, training: bool = False) -> pd.DataFrame:
    """Pipeline complet : nettoyage -> features -> normalisation."""
    cleaned = clean(df, training=training)
    with_features = add_features(cleaned)
    return normalize(with_features, scaler)
