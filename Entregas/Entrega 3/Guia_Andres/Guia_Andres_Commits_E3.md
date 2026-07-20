# Guía de trabajo — Entrega Final (Entrega 3)

Proyecto Telco Churn · Despliegue de Soluciones Analíticas en la Nube (AWS)
Para: Andrés Mauricio Rueda Blanco

**Objetivo:** hacer dos (2) commits con cambios reales al proyecto y subirlos (push) al repositorio en GitHub, de forma que queden registrados a tu nombre. Al final debes tomar una captura de pantalla que muestre tus commits y enviarla a Ferney para incluirla en el reporte de la Entrega 3.

**Contexto rápido:** desde la última vez que tocaste el repo, se agregó una API (`churn-api`, FastAPI) que empaqueta el modelo y la sirve por HTTP, y el tablero (`churn-tablero`) ahora la consume en vez de cargar el modelo directamente. Todo corre en contenedores Docker. No necesitas instalar Docker ni desplegar nada — tus dos cambios son puntuales y se prueban con Python normal, sin contenedores.

## Qué necesitas antes de empezar

- Git instalado en tu computador.
- Python 3.10 o superior instalado.
- Tu acceso de colaborador al repositorio (el mismo de la Entrega 2).

## Archivos que vas a modificar

| Archivo | Qué le vas a cambiar |
|---|---|
| `churn-api/app/model.py` | Agregar una función `model_info()` que devuelve los metadatos del modelo (tipo, hiperparámetros). |
| `churn-api/app/api.py` | Agregar un endpoint `GET /api/v1/model-info` que expone esa función. |
| `README.md` | Agregar una sección "Arquitectura (Entrega 3)" explicando cómo quedaron organizados `churn-api` y `churn-tablero`. |

Los tres archivos van incluidos en este ZIP (carpeta `archivos_a_modificar/`) tal como están hoy en el repositorio, para que los compares con lo que te llegue al clonar. Los cambios los harás directamente sobre tu copia clonada del repositorio (Paso 3), no sobre estas copias sueltas.

## Paso 1 — Clonar el repositorio

```bash
git clone https://github.com/fduarteg2/ProjectChurn.git
cd ProjectChurn
```

## Paso 2 — Configurar tu identidad en git

Importante para que los commits queden registrados a tu nombre (si ya lo hiciste en la Entrega 2 y sigues en el mismo computador, puedes saltar este paso):

```bash
git config user.name "Andrés Mauricio Rueda Blanco"
git config user.email "tu_correo@ejemplo.com"
```

## Paso 3 — Preparar un entorno mínimo

Para estos dos cambios NO necesitas scikit-learn, DVC, ni el modelo entrenado — solo dos librerías livianas:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install joblib pandas
```

---

## Commit 1 — Endpoint de información del modelo

Abre `churn-api/app/model.py`. Al final del archivo (después de la función `feature_importances`), agrega:

```python
MODEL_TYPE = "RandomForestClassifier"
N_ESTIMATORS = 400
MAX_DEPTH = 10
CLASS_WEIGHT = "balanced"


def model_info() -> dict:
    """Metadatos del modelo empaquetado (hiperparametros ganadores, Entrega 2 - MLflow)."""
    return {
        "model_type": MODEL_TYPE,
        "n_estimators": N_ESTIMATORS,
        "max_depth": MAX_DEPTH,
        "class_weight": CLASS_WEIGHT,
        "selected_in": "Entrega 2 (MLflow, run rf_n400_d10)",
    }
```

Ahora abre `churn-api/app/api.py`. Debajo del endpoint `get_feature_importances`, agrega:

```python
@api_router.get("/model-info")
def get_model_info() -> dict:
    return model.model_info()
```

**Verifica que funciona** (sin necesidad de levantar el servidor completo):

```bash
cd churn-api
python3 -c "from app.model import model_info; import json; print(json.dumps(model_info(), indent=2))"
```

Deberías ver algo como:

```json
{
  "model_type": "RandomForestClassifier",
  "n_estimators": 400,
  "max_depth": 10,
  "class_weight": "balanced",
  "selected_in": "Entrega 2 (MLflow, run rf_n400_d10)"
}
```

Guarda y sube el cambio (desde la raíz del repo, `cd ..` si estás dentro de `churn-api`):

```bash
cd ..
git add churn-api/app/model.py churn-api/app/api.py
git commit -m "Agrega endpoint /api/v1/model-info con metadatos del modelo"
git push origin main
```

## Commit 2 — Documentar la arquitectura en el README

Abre `README.md` (donde ya está tu sección de Reproducibilidad de la Entrega 2) y agrega al final:

```markdown
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
```

Guarda y sube el cambio:

```bash
git add README.md
git commit -m "Documenta la arquitectura API + tablero en el README"
git push origin main
```

## Paso 4 — Captura de evidencia

Toma **una sola captura** que muestre claramente tus dos commits, por ejemplo el resultado de:

```bash
git log --oneline -5
```

o directamente la página `https://github.com/fduarteg2/ProjectChurn/commits/main` mostrando tus dos commits con tu nombre y fecha.

**Envíale la captura a Ferney** por el mismo medio que te compartió este ZIP — él la incluye en el reporte final y actualiza la tabla de commits y el reporte de trabajo en equipo.

**Nota:** si al hacer `git push` te pide usuario y contraseña, GitHub ya no acepta la contraseña de tu cuenta directamente — necesitas un *Personal Access Token* (se genera en GitHub → Settings → Developer settings → Personal access tokens) o tener configurado SSH. Si te pasa, avísale a Ferney.

Guía preparada para la Entrega Final del proyecto Telco Churn — Despliegue de Soluciones Analíticas en la Nube (AWS).
