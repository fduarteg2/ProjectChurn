"""
Entrenamiento del modelo FINAL seleccionado para el tablero.
Proyecto: Telco Churn - Entrega 2

Este script entrena el modelo ganador (Random Forest, 400 árboles, profundidad 10)
sobre TODO el dataset disponible (no solo el split de entrenamiento) y lo guarda
en disco para que el tablero de Streamlit lo cargue directamente.

Ejecutar una sola vez (o cada vez que cambien los datos/features):
    python train_final_model.py
"""

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = "Telco-Churn.csv"
OUTPUT_MODEL_PATH = "modelo_churn_final.joblib"
RANDOM_STATE = 42

# Hiperparámetros ganadores (seleccionados en MLflow: rf_n400_d10)
N_ESTIMATORS = 400
MAX_DEPTH = 10


def load_and_clean_data(path):
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df  # conserva customerID (se usa en el tablero para seleccionar clientes)


def build_preprocessor(df):
    target = "Churn"
    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    excluded = numeric_features + [target, "customerID"]
    categorical_features = [c for c in df.columns if c not in excluded]

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


def main():
    df = load_and_clean_data(DATA_PATH)
    preprocessor, num_feats, cat_feats = build_preprocessor(df)

    X = df.drop(columns=["Churn", "customerID"])
    y = df["Churn"]

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])

    print("Entrenando modelo final sobre el dataset completo...")
    model.fit(X, y)

    joblib.dump(model, OUTPUT_MODEL_PATH)
    print(f"Modelo guardado en: {OUTPUT_MODEL_PATH}")

    # Guardamos también el dataset ya limpio (para la Matriz de Priorización del tablero)
    df.to_csv("telco_churn_clean.csv", index=False)
    print("Dataset limpio guardado en: telco_churn_clean.csv")


if __name__ == "__main__":
    main()
