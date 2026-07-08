"""
Entrenamiento y comparación de modelos de predicción de fuga (Churn) - Telco Customer Churn
Proyecto: Despliegue de Soluciones en la Nube (AWS)
Entrega 2

Este script:
1. Carga y limpia el dataset Telco-Churn.
2. Construye un pipeline de preprocesamiento (imputación, escalado, one-hot encoding).
3. Entrena y compara 3 familias de modelos con distintas variaciones de hiperparámetros:
   - Regresión Logística (baseline interpretable)
   - Random Forest
   - Gradient Boosting
4. Registra cada corrida (parámetros, métricas, modelo) en MLflow.

Antes de ejecutar, configura la variable MLFLOW_TRACKING_URI más abajo con la IP pública
de tu instancia EC2.
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURACIÓN — AJUSTA ESTO ANTES DE CORRER
# ============================================================
MLFLOW_TRACKING_URI = "http://98.93.41.190:8050"   # <-- reemplaza con tu IP pública
EXPERIMENT_NAME = "telco_churn_entrega2"
DATA_PATH = "Telco-Churn.csv"                          # <-- ajusta la ruta si es necesario
RANDOM_STATE = 42

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


# ============================================================
# 1. CARGA Y LIMPIEZA DE DATOS
# ============================================================
def load_and_clean_data(path):
    df = pd.read_csv(path)

    # TotalCharges viene como texto; hay 11 registros vacíos, todos con tenure=0
    # (clientes nuevos sin facturación acumulada). Se imputan como 0.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # customerID no aporta señal predictiva, se descarta
    df = df.drop(columns=["customerID"])

    # Variable objetivo a binaria
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


def build_preprocessor(df):
    target = "Churn"
    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_features = [
        c for c in df.columns if c not in numeric_features + [target]
    ]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])

    return preprocessor, numeric_features, categorical_features


def eval_and_log(model, X_test, y_test, run_name, params, plots_dir="plots"):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    for k, v in params.items():
        mlflow.log_param(k, v)
    for k, v in metrics.items():
        mlflow.log_metric(k, v)

    # Matriz de confusión como artefacto
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Churn", "Churn"])
    ax.set_yticklabels(["No Churn", "Churn"])
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
    ax.set_title(f"Matriz de Confusión - {run_name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.tight_layout()
    cm_path = f"{plots_dir}/{run_name}_confusion_matrix.png"
    fig.savefig(cm_path)
    plt.close(fig)
    mlflow.log_artifact(cm_path)

    mlflow.sklearn.log_model(model, "model")

    print(f"[{run_name}] " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    return metrics


def main():
    import os
    os.makedirs("plots", exist_ok=True)

    df = load_and_clean_data(DATA_PATH)
    preprocessor, num_feats, cat_feats = build_preprocessor(df)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    results = []

    # --------------------------------------------------------
    # Familia 1: Regresión Logística (baseline interpretable)
    # --------------------------------------------------------
    for C in [0.01, 0.1, 1, 10]:
        run_name = f"logreg_C{C}"
        with mlflow.start_run(run_name=run_name):
            params = {"model_type": "LogisticRegression", "C": C, "class_weight": "balanced"}
            clf = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(
                    C=C, class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
                )),
            ])
            clf.fit(X_train, y_train)
            metrics = eval_and_log(clf, X_test, y_test, run_name, params)
            results.append({"run": run_name, **params, **metrics})

    # --------------------------------------------------------
    # Familia 2: Random Forest
    # --------------------------------------------------------
    rf_configs = [
        {"n_estimators": 200, "max_depth": 5},
        {"n_estimators": 400, "max_depth": 10},
        {"n_estimators": 400, "max_depth": None},
    ]
    for cfg in rf_configs:
        run_name = f"rf_n{cfg['n_estimators']}_d{cfg['max_depth']}"
        with mlflow.start_run(run_name=run_name):
            params = {"model_type": "RandomForest", "class_weight": "balanced", **cfg}
            clf = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("classifier", RandomForestClassifier(
                    n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
                    class_weight="balanced", random_state=RANDOM_STATE
                )),
            ])
            clf.fit(X_train, y_train)
            metrics = eval_and_log(clf, X_test, y_test, run_name, params)
            results.append({"run": run_name, **params, **metrics})

    # --------------------------------------------------------
    # Familia 3: Gradient Boosting
    # --------------------------------------------------------
    gb_configs = [
        {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
        {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3},
    ]
    for cfg in gb_configs:
        run_name = f"gb_n{cfg['n_estimators']}_lr{cfg['learning_rate']}"
        with mlflow.start_run(run_name=run_name):
            params = {"model_type": "GradientBoosting", **cfg}
            clf = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("classifier", GradientBoostingClassifier(
                    n_estimators=cfg["n_estimators"], learning_rate=cfg["learning_rate"],
                    max_depth=cfg["max_depth"], random_state=RANDOM_STATE
                )),
            ])
            clf.fit(X_train, y_train)
            metrics = eval_and_log(clf, X_test, y_test, run_name, params)
            results.append({"run": run_name, **params, **metrics})

    # --------------------------------------------------------
    # Resumen comparativo en consola
    # --------------------------------------------------------
    results_df = pd.DataFrame(results)
    print("\n=== Resumen comparativo (ordenado por F1) ===")
    print(results_df.sort_values("f1_score", ascending=False)[
        ["run", "model_type", "accuracy", "precision", "recall", "f1_score", "roc_auc"]
    ].to_string(index=False))

    results_df.to_csv("plots/resumen_experimentos.csv", index=False)
    print("\nResumen guardado en plots/resumen_experimentos.csv")
    print(f"Revisa los resultados completos en la UI de MLflow: {MLFLOW_TRACKING_URI}")


if __name__ == "__main__":
    main()
