# Guion detallado — Presentación Final TelcoChurn

Para estudiar antes de la exposición (máx. 10 minutos, jueves 23 de julio). Cada bloque trae: tiempo sugerido, qué decir (guion natural, no leer literal — apropiárselo), y "para profundizar" con datos que **no están en el slide** pero que demuestran dominio si el profesor pregunta o si hay tiempo de sobra.

**Tiempo total objetivo: 9-10 min.** Practiquen con cronómetro al menos una vez completos.

---

## Slide 1 — Portada (20 seg)

**Decir:** "Buenas tardes, somos Ferney Duarte y Andrés Rueda, y les vamos a presentar TelcoChurn: un sistema de predicción de fuga de clientes que construimos y desplegamos como un producto completo — modelo, API y tablero — corriendo en la nube."

**Para profundizar (si preguntan por qué el nombre / el dataset):** El dataset es "Telco Customer Churn" de Kaggle, ampliamente usado como benchmark académico para problemas de churn en telecomunicaciones — lo elegimos porque tiene una mezcla realista de variables demográficas, de servicios y de facturación, y un desbalance de clases moderado (no extremo), que permite comparar estrategias de rebalanceo sin resultados triviales.

---

## Slide 2 — Agenda (15 seg)

**Decir:** "Vamos a cubrir seis puntos: el problema de negocio, los modelos que construimos, la solución de tablero más API, cómo la desplegamos, una demo en vivo, y cerramos con resultados y conclusiones."

**Para profundizar:** No es necesario detenerse aquí — es solo un mapa. Pasen rápido.

---

## Slide 3 — El problema (60 seg)

**Decir:** "El problema de negocio parte de una asimetría bien conocida: retener un cliente cuesta mucho menos que adquirir uno nuevo. Pero hoy los equipos de gestión de cuentas actúan de forma reactiva — se enteran de que un cliente está insatisfecho cuando ya canceló. Nuestra pregunta de negocio es: ¿qué clientes tienen mayor probabilidad de cancelar en el corto plazo, y qué factores explican esa decisión, para poder priorizar el esfuerzo comercial? Trabajamos con 7,043 clientes y 20 variables predictoras — demografía, servicios contratados, y datos de cuenta y pagos."

**Para profundizar:**
- La tasa de churn histórica en el dataset es ~26.5% — un desbalance real pero manejable (no es un caso de 1% de fraude, por ejemplo).
- Las 20 variables se agrupan en 4 bloques: identificación (`customerID`, descartado), demografía (`gender`, `SeniorCitizen`, `Partner`, `Dependents`), servicios contratados (teléfono, internet, seguridad online, soporte técnico, streaming) y cuenta/pagos (`Contract`, `PaperlessBilling`, `PaymentMethod`, `tenure`, `MonthlyCharges`, `TotalCharges`).
- No hubo cambios de alcance respecto a la Entrega 2 — el foco de esta entrega fue el despliegue en producción (API + Docker), no el modelado.

---

## Slide 4 — Modelos construidos (60-70 seg)

**Decir:** "Construimos y comparamos 9 experimentos distintos, en 3 familias de modelos: Regresión Logística como línea base interpretable, Random Forest, y Gradient Boosting. Todo esto lo registramos en MLflow, corriendo en un servidor propio en AWS EC2 — cada corrida quedó con sus hiperparámetros, sus métricas y el modelo serializado como artefacto. En el gráfico ven el F1-score de las 9 corridas: el mejor resultado lo dio un Random Forest con 400 árboles y profundidad máxima 10."

**Para profundizar:**
- Preprocesamiento: pipeline de scikit-learn con `ColumnTransformer` — las 3 variables numéricas (`tenure`, `MonthlyCharges`, `TotalCharges`) se imputan por mediana y se escalan con `StandardScaler`; las 16 categóricas se imputan por moda y se codifican con `OneHotEncoder`. Split 80/20 estratificado, `random_state=42` para reproducibilidad.
- `TotalCharges` venía como texto en el CSV original (11 registros vacíos, correspondientes a clientes con `tenure=0`, es decir, recién llegados sin facturación acumulada) — se convirtió a numérico y se imputó como 0.
- Regresión Logística usó `class_weight="balanced"` con 4 valores de C (0.01, 0.1, 1, 10) para explorar regularización.
- Random Forest probó 3 combinaciones de `n_estimators`/`max_depth`, también con `class_weight="balanced"`.
- Gradient Boosting probó 2 combinaciones de `n_estimators`/`learning_rate`/`max_depth`.
- El servidor de MLflow corrió en una instancia EC2 (`mlflow-churn-server`), con el proceso persistido vía `systemd` para sobrevivir a desconexiones SSH — eso fue trabajo específico de infraestructura, no solo lanzar el proceso a mano.

---

## Slide 5 — Modelo seleccionado (50-60 seg)

**Decir:** "El modelo ganador es un Random Forest — 400 árboles, profundidad máxima 10, con balanceo de clases. Tiene el mejor F1-score del conjunto, 0.622, con un recall de 0.72: es decir, detecta el 72% de los clientes que realmente se van a ir. Elegimos priorizar recall sobre precisión porque, en este caso de negocio, el costo de no detectar a un cliente que se va es mayor que el costo de una alerta de más — un Account Manager simplemente revisa una lista más larga, pero no perdemos clientes por no anticiparlos."

**Para profundizar:**
- Comparación explícita: Regresión Logística tiene el recall más alto de todos (~0.78-0.79) pero precisión muy baja (~0.50) — es un modelo "sobre-alertador". Gradient Boosting tiene la mejor accuracy (0.80) y mejor precisión (0.67) pero su recall cae a ~0.52 — deja pasar casi la mitad de los churners reales.
- Random Forest sin límite de profundidad (`max_depth=None`) da más accuracy aparente (0.78) pero el recall se desploma a 0.47 — es un caso claro de sobreajuste, y es la evidencia que usamos para justificar limitar la profundidad a 10.
- ROC AUC del modelo elegido: 0.840 — competitivo con el mejor de todos los experimentos (Gradient Boosting, 0.844), pero con muchísimo mejor recall.
- El modelo final se reentrena sobre el 100% de los datos (no solo el split de entrenamiento) en `train_final_model.py`, y se serializa con `joblib` — ese archivo (`modelo_churn_final.joblib`, ~27 MB) es el que carga la API en producción.

---

## Slide 6 — La solución: tablero + API (50 seg)

**Decir:** "Con el modelo ya elegido, construimos un producto completo: una API que sirve las predicciones, y un tablero interactivo en Streamlit con 3 vistas. El Simulador de Escenarios permite ajustar variables de un cliente y ver en tiempo real cómo cambia su probabilidad de fuga. La Matriz de Priorización cruza riesgo contra valor del cliente para identificar la 'Zona Crítica' — a quién atender primero. Y Factores de Influencia da explicabilidad: qué variables pesan más en la predicción del modelo."

**Para profundizar:**
- La Matriz de Priorización usa `MonthlyCharges` como proxy de valor del cliente (no hay un campo de "valor de vida del cliente" explícito en el dataset, así que se usó la facturación mensual como aproximación razonable y ajustable).
- Los Factores de Influencia se calculan con la importancia nativa (Gini importance) del Random Forest, agregada por variable de negocio original — no por cada categoría del one-hot encoding, para que sea legible para un usuario de negocio.
- Es consistente end-to-end: los mismos 3 factores más importantes (`Contract`, `tenure`, `TotalCharges`) aparecieron también en el EDA exploratorio de la Entrega 1 — eso valida que el modelo aprendió señal real de negocio, no ruido.

---

## Slide 7 — Arquitectura: API + Tablero + Docker (70-80 seg — el corazón técnico de esta entrega)

**Decir:** "Esta es la novedad central de la Entrega Final. Hasta la entrega anterior, el tablero cargaba el modelo directamente con `joblib`. Para esta entrega separamos esa responsabilidad: construimos `churn-api`, un servicio FastAPI que empaqueta el modelo y expone endpoints REST — salud, predicción individual, predicción en lote, e importancia de variables. El tablero ahora es un cliente HTTP de esa API, no tiene el modelo embebido. Ambos servicios corren en contenedores Docker independientes, orquestados con `docker compose`, desplegados en una instancia AWS EC2 dedicada."

**Para profundizar (esta es la parte donde más pueden preguntar):**
- Por qué separar: desacopla el ciclo de vida del modelo del de la interfaz — se puede actualizar el modelo y redesplegar solo `churn-api` sin tocar el tablero, y viceversa. Es la arquitectura que se usaría en un escenario de producción real con varios consumidores de un mismo modelo (podría haber otro cliente además del tablero — un app móvil, otro sistema interno — consumiendo la misma API).
- Los 2 contenedores se comunican por la red interna que crea `docker compose` automáticamente: el tablero llama a `http://churn-api:8001` usando el nombre del servicio como hostname (resolución DNS interna de Docker), no una IP fija.
- Puertos: `churn-api` en 8001 (FastAPI/Uvicorn), `churn-tablero` en 8501 (Streamlit).
- Detalle de infraestructura: el modelo (`modelo_churn_final.joblib`) y el dataset limpio no viajan por Git — se regeneran en la máquina de despliegue a partir del dataset versionado en DVC (remoto S3), corriendo `dvc pull` + `train_final_model.py` antes de construir las imágenes Docker. Esto mantiene el repositorio liviano y totalmente reproducible.
- Usamos una instancia EC2 **distinta** a la del servidor de MLflow, específicamente para no arriesgar la evidencia de los experimentos de la Entrega 2.
- Endpoints exactos de la API: `GET /health`, `POST /predict`, `POST /predict_batch` (usado por la Matriz de Priorización para anotar hasta 1,500 clientes en una sola llamada), `GET /feature-importances`.

---

## Slide 8 — Demo: la API en producción (40-50 seg)

**Decir:** "Aquí ven la documentación interactiva de la API — Swagger, autogenerada por FastAPI — corriendo en la IP pública de nuestra instancia EC2. Le enviamos el perfil de un cliente al endpoint `/predict` y responde con la probabilidad de fuga y la etiqueta de riesgo. Esto confirma que no es una demo local: la API está viva en la nube."

**Para profundizar:** Si hay tiempo, mencionar que también expusimos `/docs` (Swagger) sin necesidad de escribir documentación manual — es un beneficio directo de usar FastAPI con Pydantic: los esquemas de request/response se derivan del código y quedan documentados automáticamente.

---

## Slide 9 — Demo: el tablero, Simulador de Escenarios (40-50 seg)

**Decir:** "Y aquí el tablero, ya consumiendo esa misma API. Un Account Manager elige un cliente, cambia por ejemplo el tipo de contrato o si tiene soporte técnico, y ve al instante cómo cambia la probabilidad de fuga — en este caso, 62%, alto riesgo. Cada cambio dispara una llamada HTTP real a `churn-api`, no hay cómputo local en el tablero."

**Para profundizar:** El "cambio (delta)" que se muestra junto al resultado compara la probabilidad simulada contra la probabilidad del perfil original del cliente — eso ayuda a cuantificar el impacto de una acción de retención específica (p. ej., "ofrecerle un contrato anual reduciría su riesgo en X puntos porcentuales").

---

## Slide 10 — Demo: priorización y explicabilidad (40-50 seg)

**Decir:** "Por un lado, la Matriz de Priorización: cada punto es un cliente, cruzando probabilidad de fuga contra facturación mensual. Los puntos rojos, en la esquina de alto riesgo y alto valor, son la Zona Crítica — en esta muestra, 395 clientes que el equipo de gestión debería atender primero. Y por otro lado, Factores de Influencia, que confirma que `Contract`, antigüedad y facturación total son las variables que más pesan en la predicción."

**Para profundizar:** Ambos umbrales (de riesgo y de valor) son ajustables con sliders en vivo — el tablero no tiene un único punto de corte fijo, se adapta a la política de negocio que quiera aplicar cada equipo comercial.

---

## Slide 11 — Resultados y conclusiones (50-60 seg)

**Decir:** "En resumen: entregamos un prototipo funcional completo — modelo, API y tablero — desplegado en la nube y accesible públicamente, no solo un notebook. La arquitectura desacoplada nos acerca a un escenario de producción real. El modelo cumple el objetivo de negocio: identifica automáticamente las cuentas que más urge atender. Y los factores más importantes son consistentes desde el análisis exploratorio de la primera entrega hasta el modelo final, lo cual nos da confianza en que el modelo capturó señal real y no ruido. Como trabajo futuro, dejamos identificado: autenticación en la API, monitoreo de drift del modelo en producción, y un pipeline de CI/CD que dispare automáticamente un nuevo despliegue ante cada actualización del modelo o del tablero."

**Para profundizar:** Si preguntan qué es "drift": es el fenómeno donde el comportamiento de los clientes cambia con el tiempo (nuevas ofertas de la competencia, cambios económicos) y el modelo entrenado con datos históricos pierde precisión — monitorearlo implica comparar la distribución de las predicciones/datos de entrada en producción contra la distribución de entrenamiento.

---

## Slide 12 — Cierre (10 seg)

**Decir:** "Gracias. Quedamos atentos a sus preguntas y retroalimentación."

---

## Preguntas frecuentes que podrían hacerles (preparar respuesta corta)

1. **¿Por qué Random Forest y no Gradient Boosting si tiene mejor accuracy?** → Por el costo de negocio asimétrico: un falso negativo (cliente que se va y no lo detectamos) es más caro que un falso positivo (alerta de más). Recall es la métrica que más nos importa, y ahí Random Forest gana claramente (0.72 vs 0.52).
2. **¿Por qué separar el tablero de la API si funcionaba igual antes?** → Desacople de ciclo de vida, reutilización del modelo por otros consumidores, y es el patrón estándar de arquitectura en producción (microservicios).
3. **¿Qué pasa si la API se cae?** → El tablero detecta el error de conexión y muestra un mensaje claro al usuario en vez de fallar silenciosamente (lo implementamos explícitamente con un chequeo de `/health` al cargar el tablero).
4. **¿Los datos están seguros?** → El dataset se versiona en DVC con un remoto S3 privado (acceso público bloqueado); las credenciales de AWS usadas son temporales (AWS Academy Learner Lab).
5. **¿Cómo se actualiza el modelo en producción?** → Se reentrena con `train_final_model.py`, se reconstruye solo la imagen de `churn-api` (`docker compose up -d --build churn-api`), y el tablero sigue funcionando sin cambios porque solo depende del contrato de la API, no del modelo internamente.
