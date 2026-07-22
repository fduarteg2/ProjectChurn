# ProjectChurn


## Reproducibilidad
1. Clonar el repositorio y crear un entorno virtual:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_local.txt
```
2. Obtener los datos versionados con DVC:
```bash dvc pull ```
3. Entrenar y comparar modelos, registrando experimentos en MLflow:
```bash python train_churn_models.py ```
4. Entrenar el modelo final para el tablero:
```bash python train_final_model.py ```


## Arquitectura (Entrega 3)
A partir de la Entrega 3, la solución se compone de dos servicios independientes,
cada uno en su propio contenedor Docker:
- **churn-api** (FastAPI, puerto 8001): empaqueta `modelo_churn_final.joblib` y
  expone `/health`, `/predict`, `/predict_batch`, `/feature-importances` y
  `/model-info`.
- **churn-tablero** (Streamlit, puerto 8501): ya no carga el modelo directamente,
  consume `churn-api` por HTTP (variables de entorno `API_URL` / `API_PORT`).
Ambos se levantan juntos con:
\`\`\`bash
docker compose build
docker compose up -d
\`\`\`
Ver `Entregas/Entrega 3/Manual_Instalacion_Tablero.md` para el detalle completo.