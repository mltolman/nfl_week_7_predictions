from __future__ import annotations
import os
import joblib
import pandas as pd
from typing import Dict, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score

def time_based_split(df: pd.DataFrame, test_start_season: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["season"] < test_start_season].copy()
    test  = df[df["season"] >= test_start_season].copy()
    return train, test

def train_logistic(X_train, y_train) -> Tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xs, y_train)
    return scaler, clf

def evaluate_model(model, scaler, X, y) -> Dict[str, float]:
    Xs = scaler.transform(X)
    p = model.predict_proba(Xs)[:,1]
    pred = (p >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "accuracy": float(accuracy_score(y, pred)),
    }

def save_artifacts(model_dir: str, scaler: StandardScaler, model: LogisticRegression):
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(model_dir, "scaler.joblib"))
    joblib.dump(model, os.path.join(model_dir, "model.joblib"))