"""
Tablero de Predicción de Fuga (Churn) - Telco Customer Churn
Proyecto: Despliegue de Soluciones en la Nube (AWS) - Entrega 2

Este tablero consume directamente el modelo entrenado (modelo_churn_final.joblib)
sin pasar por una API separada (alcance acordado para la Entrega 2).

Ejecutar con:
    streamlit run tablero.py
"""

import joblib
import pandas as pd
import numpy as np
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

MODEL_PATH = "modelo_churn_final.joblib"
DATA_PATH = "telco_churn_clean.csv"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


@st.cache_data
def score_all_customers(_model, df):
    """Calcula la probabilidad de fuga para todos los clientes (para la Matriz de Priorización)."""
    X = df.drop(columns=["Churn", "customerID"])
    proba = _model.predict_proba(X)[:, 1]
    result = df.copy()
    result["Probabilidad_Fuga"] = proba
    return result


def get_feature_importances(model):
    """Agrega las importancias del Random Forest por variable de negocio original
    (no por categoría individual del one-hot encoding)."""
    pre = model.named_steps["preprocessor"]
    clf = model.named_steps["classifier"]

    num_features = pre.transformers_[0][2]
    cat_features = pre.transformers_[1][2]
    onehot = pre.transformers_[1][1].named_steps["onehot"]

    importances = clf.feature_importances_
    idx = 0
    agg = {}
    for f in num_features:
        agg[f] = importances[idx]
        idx += 1
    for f, cats in zip(cat_features, onehot.categories_):
        n = len(cats)
        agg[f] = importances[idx:idx + n].sum()
        idx += n

    s = pd.Series(agg).sort_values(ascending=False)
    return s


model = load_model()
df = load_data()
scored_df = score_all_customers(model, df)

st.title("📊 Panel de Retención de Clientes — TelcoChurn")
st.caption(
    "MVP de predicción de fuga (Churn) para priorizar acciones de retención. "
    "Entrega 2 — Despliegue de Soluciones en la Nube."
)

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
        "para ver cómo cambia su probabilidad de fuga en tiempo real."
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

        X_sim = pd.DataFrame([sim_row.drop(["Churn", "customerID"])])
        proba_sim = model.predict_proba(X_sim)[0, 1]

        X_base = pd.DataFrame([base_row.drop(["Churn", "customerID"])])
        proba_base = model.predict_proba(X_base)[0, 1]

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
        "Cruce entre la probabilidad de fuga predicha y la facturación mensual (MRR), "
        "para identificar cuentas de **Zona Crítica** (alto riesgo + alto valor)."
    )

    threshold_risk = st.slider("Umbral de Alto Riesgo (probabilidad)", 0.0, 1.0, 0.5, 0.05)
    threshold_value = st.slider(
        "Umbral de Alto Valor (MonthlyCharges, USD)",
        float(scored_df["MonthlyCharges"].min()),
        float(scored_df["MonthlyCharges"].max()),
        float(scored_df["MonthlyCharges"].median()),
        5.0,
    )

    plot_df = scored_df.sample(min(1500, len(scored_df)), random_state=42)  # muestra para rendimiento visual

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

    plot_df = plot_df.copy()
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
        "Random Forest (agregada por variable original, no por categoría)."
    )

    importances = get_feature_importances(model)
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
        "no un análisis SHAP. Es coherente con los hallazgos del EDA de la Entrega 1: "
        "Contrato, antigüedad (tenure) y facturación son los factores más determinantes."
    )
