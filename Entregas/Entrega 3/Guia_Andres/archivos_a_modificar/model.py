"""Carga del modelo empaquetado y logica de inferencia / explicabilidad.

El modelo (modelo_churn_final.joblib) es un Pipeline de scikit-learn
(preprocessor + RandomForestClassifier) entrenado en train_final_model.py.
"""

import os

import joblib
import pandas as pd

MODEL_PATH = os.environ.get("MODEL_PATH", "modelo_churn_final.joblib")
RISK_THRESHOLD = 0.5

_model = None


def load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def is_model_loaded() -> bool:
    return _model is not None


def _to_dataframe(customers: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(customers)


def predict_one(customer: dict) -> dict:
    model = load_model()
    X = _to_dataframe([customer])
    proba = float(model.predict_proba(X)[0, 1])
    label = "ALTO RIESGO" if proba >= RISK_THRESHOLD else "BAJO RIESGO"
    return {"churn_probability": proba, "risk_label": label}


def predict_many(customers: list[dict]) -> list[dict]:
    model = load_model()
    X = _to_dataframe(customers)
    probas = model.predict_proba(X)[:, 1]
    return [
        {
            "churn_probability": float(p),
            "risk_label": "ALTO RIESGO" if p >= RISK_THRESHOLD else "BAJO RIESGO",
        }
        for p in probas
    ]


def feature_importances() -> list[dict]:
    """Agrega las importancias del Random Forest por variable de negocio original
    (no por categoria individual del one-hot encoding)."""
    model = load_model()
    pre = model.named_steps["preprocessor"]
    clf = model.named_steps["classifier"]

    num_features = pre.transformers_[0][2]
    cat_features = pre.transformers_[1][2]
    onehot = pre.transformers_[1][1].named_steps["onehot"]

    importances = clf.feature_importances_
    idx = 0
    agg = {}
    for f in num_features:
        agg[f] = float(importances[idx])
        idx += 1
    for f, cats in zip(cat_features, onehot.categories_):
        n = len(cats)
        agg[f] = float(importances[idx:idx + n].sum())
        idx += n

    ordered = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)
    return [{"feature": f, "importance": v} for f, v in ordered]
