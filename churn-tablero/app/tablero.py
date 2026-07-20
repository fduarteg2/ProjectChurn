"""
Tablero de Prediccion de Fuga (Churn) - Telco Customer Churn
Proyecto: Despliegue de Soluciones en la Nube (AWS) - Entrega 3

A partir de la Entrega 3, el tablero YA NO carga el modelo directamente:
consume las predicciones y la explicabilidad a traves de la API churn-api
(POST /api/v1/predict, /api/v1/predict_batch, GET /api/v1/feature-importances),
configurable mediante las variables de entorno API_URL y API_PORT.

Ejecutar con:
    streamlit run tablero.py
"""

import os

import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# CONFIGURACIÓN VISUAL (misma paleta corporativa del EDA)
# ============================================================
COLOR_NAVY = "#1F4E79"
COLOR_CHARCOAL = "#363636"
COLOR_CORAL = "#F16A70"
COLOR_LIGHTBLUE = "#8ECAE6"

st.set_page_config(
    page_title="TelcoChurn | Panel de Retención",
    page_icon="📊",
    layout="wide",
)

st.markdown(f"""
<style>
    .stApp {{ background-color: #F7F9FB; }}
    h1, h2, h3 {{ color: {COLOR_NAVY}; }}
    .risk-box {{
        padding: 1.2rem; border-radius: 10px; text-align: center;
        color: white; font-weight: bold;
    }}
</style>
""", unsafe_allow_html=True)

DATA_PATH = "telco_churn_clean.csv"
API_URL = os.environ.get("API_URL", "localhost")
API_PORT = os.environ.get("API_PORT", "8001")
API_BASE = f"http://{API_URL}:{API_PORT}/api/v1"

MODEL_FEATURE_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]


def row_to_customer_payload(row: pd.Series) -> dict:
    """Convierte una fila del dataset al esquema CustomerFeatures esperado por la API."""
    return {
        "gender": str(row["gender"]),
        "SeniorCitizen": int(row["SeniorCitizen"]),
        "Partner": str(row["Partner"]),
        "Dependents": str(row["Dependents"]),
        "tenure": int(row["tenure"]),
        "PhoneService": str(row["PhoneService"]),
        "MultipleLines": str(row["MultipleLines"]),
        "InternetService": str(row["InternetService"]),
        "OnlineSecurity": str(row["OnlineSecurity"]),
        "OnlineBackup": str(row["OnlineBackup"]),
        "DeviceProtection": str(row["DeviceProtection"]),
        "TechSupport": str(row["TechSupport"]),
        "StreamingTV": str(row["StreamingTV"]),
        "StreamingMovies": str(row["StreamingMovies"]),
        "Contract": str(row["Contract"]),
        "PaperlessBilling": str(row["PaperlessBilling"]),
        "PaymentMethod": str(row["PaymentMethod"]),
        "MonthlyCharges": float(row["MonthlyCharges"]),
        "TotalCharges": float(row["TotalCharges"]),
    }


def api_predict_one(row: pd.Series) -> float:
    payload = {"customer": row_to_customer_payload(row)}
    resp = requests.post(f"{API_BASE}/predict", json=payload, timeout=15)
    resp.raise_for_status()
    return resp.json()["churn_probability"]


@st.cache_data(show_spinner="Consultando predicciones a la API...")
def score_sample_via_api(_api_base: str, sample_df: pd.DataFrame) -> pd.DataFrame:
    customers = [row_to_customer_payload(row) for _, row in sample_df.iterrows()]
    resp = requests.post(f"{_api_base}/predict_batch", json={"customers": customers}, timeout=60)
    resp.raise_for_status()
    probs = [p["churn_probability"] for p in resp.json()["predictions"]]
    result = sample_df.copy()
    result["Probabilidad_Fuga"] = probs
    return result


@st.cache_data(show_spinner="Consultando importancia de variables a la API...")
def get_feature_importances_via_api(_api_base: str) -> pd.Series:
    resp = requests.get(f"{_api_base}/feature-importances", timeout=30)
    resp.raise_for_status()
    data = resp.json()["importances"]
    s = pd.Series({d["feature"]: d["importance"] for d in data})
    return s.sort_values(ascending=False)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


df = load_data()

st.title("📊 Panel de Retención de Clientes — TelcoChurn")
st.caption(
    "MVP de predicción de fuga (Churn) para priorizar acciones de retención. "
    "Entrega 3 — Tablero + API desplegados en contenedores Docker."
)

try:
    health = requests.get(f"{API_BASE}/health", timeout=5)
    health.raise_for_status()
except requests.exceptions.RequestException as exc:
    st.error(
        f"No se pudo conectar con la API de predicción en `{API_BASE}`. "
        f"Verifica que el contenedor churn-api esté corriendo. Detalle: {exc}"
    )
    st.stop()

tab1, tab2, tab3 = st.tabs([
    "🎯 Simulador de Escenarios",
    "🗺️ Matriz de Priorización",
    "🔍 Factores de Influencia",
])

# ============================================================
# TAB 1 — SIMULADOR DE ESCENARIOS
# ============================================================
with tab1:
    st.subheader("Simulador de Escenarios")
    st.write(
        "Selecciona un cliente base y ajusta variables clave del negocio "
        "para ver cómo cambia su probabilidad de fuga en tiempo real (calculada por la API)."
    )

    col_select, col_result = st.columns([1.3, 1])

    with col_select:
        customer_id = st.selectbox(
            "Cliente base (customerID)",
            options=df["customerID"].tolist(),
            index=0,
        )
        base_row = df[df["customerID"] == customer_id].iloc[0].copy()

        st.markdown("**Ajusta las variables de negocio:**")

        contract = st.selectbox(
            "Tipo de Contrato",
            options=["Month-to-month", "One year", "Two year"],
            index=["Month-to-month", "One year", "Two year"].index(base_row["Contract"]),
        )
        tech_support = st.selectbox(
            "Soporte Técnico",
            options=["No", "Yes", "No internet service"],
            index=["No", "Yes", "No internet service"].index(base_row["TechSupport"])
            if base_row["TechSupport"] in ["No", "Yes", "No internet service"] else 0,
        )
        internet_service = st.selectbox(
            "Servicio de Internet",
            options=["DSL", "Fiber optic", "No"],
            index=["DSL", "Fiber optic", "No"].index(base_row["InternetService"]),
        )
        payment_method = st.selectbox(
            "Método de Pago",
            options=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            index=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"].index(base_row["PaymentMethod"]),
        )

        # Construir el registro modificado
        sim_row = base_row.copy()
        sim_row["Contract"] = contract
        sim_row["TechSupport"] = tech_support
        sim_row["InternetService"] = internet_service
        sim_row["PaymentMethod"] = payment_method

        proba_sim = api_predict_one(sim_row)
        proba_base = api_predict_one(base_row)

    with col_result:
        risk_color = COLOR_CORAL if proba_sim >= 0.5 else COLOR_NAVY
        risk_label = "ALTO RIESGO" if proba_sim >= 0.5 else "BAJO RIESGO"

        st.markdown(f"""
        <div class="risk-box" style="background-color:{risk_color};">
            <div style="font-size:0.9rem;">Probabilidad de Fuga</div>
            <div style="font-size:3rem;">{proba_sim*100:.0f}%</div>
            <div style="font-size:1rem;">{risk_label}</div>
        </div>
        """, unsafe_allow_html=True)

        delta = (proba_sim - proba_base) * 100
        st.metric(
            label="Cambio vs. perfil original del cliente",
            value=f"{proba_sim*100:.1f}%",
            delta=f"{delta:+.1f} pts",
            delta_color="inverse",
        )

        st.write("**Perfil simulado:**")
        perfil_display = sim_row[["Contract", "TechSupport", "InternetService", "PaymentMethod",
                                   "tenure", "MonthlyCharges"]].astype(str).to_frame(name="Valor")
        st.dataframe(perfil_display, use_container_width=True)

# ============================================================
# TAB 2 — MATRIZ DE PRIORIZACIÓN
# ============================================================
with tab2:
    st.subheader("Matriz de Priorización (Riesgo vs. Valor)")
    st.write(
        "Cruce entre la probabilidad de fuga predicha (vía API) y la facturación mensual (MRR), "
        "para identificar cuentas de **Zona Crítica** (alto riesgo + alto valor)."
    )

    sample_df = df.sample(min(1500, len(df)), random_state=42)
    scored_df = score_sample_via_api(API_BASE, sample_df)

    threshold_risk = st.slider("Umbral de Alto Riesgo (probabilidad)", 0.0, 1.0, 0.5, 0.05)
    threshold_value = st.slider(
        "Umbral de Alto Valor (MonthlyCharges, USD)",
        float(scored_df["MonthlyCharges"].min()),
        float(scored_df["MonthlyCharges"].max()),
        float(scored_df["MonthlyCharges"].median()),
        5.0,
    )

    def quadrant(row):
        alto_riesgo = row["Probabilidad_Fuga"] >= threshold_risk
        alto_valor = row["MonthlyCharges"] >= threshold_value
        if alto_riesgo and alto_valor:
            return "Zona Crítica (Alto Riesgo + Alto Valor)"
        elif alto_riesgo:
            return "Alto Riesgo, Bajo Valor"
        elif alto_valor:
            return "Bajo Riesgo, Alto Valor"
        else:
            return "Bajo Riesgo, Bajo Valor"

    plot_df = scored_df.copy()
    plot_df["Cuadrante"] = plot_df.apply(quadrant, axis=1)

    color_map = {
        "Zona Crítica (Alto Riesgo + Alto Valor)": COLOR_CORAL,
        "Alto Riesgo, Bajo Valor": "#F4A6A9",
        "Bajo Riesgo, Alto Valor": COLOR_LIGHTBLUE,
        "Bajo Riesgo, Bajo Valor": COLOR_NAVY,
    }

    fig = px.scatter(
        plot_df, x="Probabilidad_Fuga", y="MonthlyCharges", color="Cuadrante",
        color_discrete_map=color_map,
        hover_data=["customerID", "Contract", "tenure"],
        labels={"Probabilidad_Fuga": "Probabilidad de Fuga", "MonthlyCharges": "Facturación Mensual (USD)"},
    )
    fig.add_vline(x=threshold_risk, line_dash="dash", line_color=COLOR_CHARCOAL)
    fig.add_hline(y=threshold_value, line_dash="dash", line_color=COLOR_CHARCOAL)
    fig.update_layout(height=550, plot_bgcolor="white", legend=dict(orientation="h", y=-0.2))

    st.plotly_chart(fig, use_container_width=True)

    n_critical = (plot_df["Cuadrante"] == "Zona Crítica (Alto Riesgo + Alto Valor)").sum()
    st.info(
        f"🔴 **{n_critical} clientes** de la muestra mostrada caen en Zona Crítica. "
        "Estas son las cuentas que el equipo de gestión debe priorizar primero."
    )

# ============================================================
# TAB 3 — FACTORES DE INFLUENCIA
# ============================================================
with tab3:
    st.subheader("Factores de Influencia del Modelo")
    st.write(
        "Importancia relativa de cada variable de negocio en la predicción del modelo "
        "Random Forest (agregada por variable original, no por categoría). Obtenida vía API."
    )

    importances = get_feature_importances_via_api(API_BASE)
    top_n = st.slider("Número de variables a mostrar", 5, 15, 8)
    top_importances = importances.head(top_n).sort_values(ascending=True)

    fig2 = go.Figure(go.Bar(
        x=top_importances.values,
        y=top_importances.index,
        orientation="h",
        marker_color=COLOR_CORAL,
    ))
    fig2.update_layout(
        height=400 + top_n * 10,
        xaxis_title="Importancia relativa",
        plot_bgcolor="white",
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.caption(
        "Nota: esta es la importancia nativa del Random Forest (Gini importance), "
        "calculada por churn-api y expuesta en GET /api/v1/feature-importances. "
        "Es coherente con los hallazgos del EDA de la Entrega 1: "
        "Contrato, antigüedad (tenure) y facturación son los factores más determinantes."
    )
