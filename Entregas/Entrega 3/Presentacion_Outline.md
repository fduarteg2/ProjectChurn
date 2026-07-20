# Presentación Final — TelcoChurn (guion, máx. 10 min)

**Sugerencia de tiempos:** ~1 min por slide en las primeras 3, ~1.5 min en modelos/arquitectura, ~2 min en demo, ~1 min conclusiones.

## Slide 1 — Portada
TelcoChurn — Predicción de Fuga de Clientes. Equipo: Ferney Duarte, Andrés Rueda. Despliegue de Soluciones Analíticas.

## Slide 2 — El problema (contexto + pregunta de negocio)
- Retener cliente < adquirir cliente nuevo. Account Managers reaccionan tarde (cuando ya canceló).
- Pregunta: ¿qué clientes tienen mayor probabilidad de fuga y qué factores la explican, para priorizar retención?
- Dataset: Telco Customer Churn (Kaggle), 7,043 clientes, 20 variables predictoras.

## Slide 3 — Modelos construidos
- 3 familias, 9 experimentos, todos registrados en MLflow (mostrar pantallazo del listado de runs).
- Modelo ganador: Random Forest (400 árboles, profundidad 10) — F1 0.622, recall 0.72, ROC AUC 0.84.
- Por qué: prioriza detectar clientes en riesgo (menos falsos negativos) sin sacrificar demasiada precisión.

## Slide 4 — Arquitectura de la solución (novedad de esta entrega)
- Antes (Entrega 2): tablero cargaba el modelo directamente.
- Ahora: `churn-api` (FastAPI) empaqueta el modelo y expone endpoints REST; `churn-tablero` (Streamlit) consume la API.
- Ambos en contenedores Docker independientes, orquestados con `docker compose`, desplegados en AWS EC2.
- (Diagrama simple: [Usuario] → [churn-tablero:8501] → [churn-api:8001] → [modelo .joblib])

## Slide 5 — Demo: API
- Mostrar Swagger UI (`/docs`) en vivo o pantallazo: POST /predict con un cliente de ejemplo → probabilidad de fuga.

## Slide 6 — Demo: Tablero (Simulador de Escenarios)
- Seleccionar cliente, cambiar contrato/soporte técnico → ver cambio de probabilidad en tiempo real.

## Slide 7 — Demo: Tablero (Matriz de Priorización + Factores de Influencia)
- Zona Crítica (alto riesgo + alto valor) — a quién priorizar primero.
- Factores más influyentes: Contract, tenure, TotalCharges.

## Slide 8 — Resultados y conclusiones
- Prototipo funcional completo: modelo + API + tablero, desplegado en la nube, accesible públicamente.
- Arquitectura desacoplada (API separada) facilita mantenimiento y actualización del modelo sin tocar el tablero.
- Conclusión de negocio: el modelo identifica automáticamente las cuentas de mayor prioridad para retención.

## Slide 9 — Trabajo futuro
- Autenticación en la API, monitoreo de drift del modelo, pipeline CI/CD para despliegue automático.

## Slide 10 — Cierre
Gracias. Preguntas.

---
**Recordatorio:** cada integrante debe dar retroalimentación a otro grupo durante la sesión (rúbrica, 5 pts) — no es delegable.
