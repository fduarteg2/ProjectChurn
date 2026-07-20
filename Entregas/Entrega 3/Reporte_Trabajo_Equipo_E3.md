# Reporte de Trabajo en Equipo — Entrega Final (Entrega 3)

**Equipo:** Ferney Duarte Garzón (202526327) — Andrés Mauricio Rueda Blanco (202524511)

| Fase | Actividad | Responsable | Evidencia |
|---|---|---|---|
| Diseño de API | Diseño y desarrollo de `churn-api` (FastAPI): endpoints `/health`, `/predict`, `/predict_batch`, `/feature-importances`, empaquetado del modelo `modelo_churn_final.joblib` | Ferney Duarte | Commit `0b4aaac` |
| Refactor del tablero | Migración de `tablero.py` de carga directa del modelo a consumo vía HTTP de `churn-api` (`churn-tablero`) | Ferney Duarte | Commit `0b4aaac` |
| Containerización | Dockerfiles de `churn-api` y `churn-tablero`, `docker-compose.yml` para orquestar ambos servicios | Ferney Duarte | Commit `0b4aaac` |
| Infraestructura | Lanzamiento de instancia AWS EC2 dedicada (`churn-app-server`), instalación de Docker, configuración de Security Group (puertos 8001/8501) | Ferney Duarte | Pantallazos de despliegue |
| Despliegue | Regeneración del modelo vía DVC (`dvc pull` + `train_final_model.py`) y build/despliegue de los contenedores en EC2 | Ferney Duarte | Pantallazos de despliegue |
| Verificación | Pruebas de la API (Swagger) y del tablero desplegado (3 vistas) contra el despliegue en la nube | Ferney Duarte | Pantallazos de despliegue |
| Documentación | Manual de Usuario del Tablero, Manual de Instalación, Reporte de la Entrega 3 | Ferney Duarte | Este documento y anexos |
| Organización del repositorio | Reorganización de la carpeta `Entregas` por entrega | Ferney Duarte | Commit `a7f21a9` |
| Modelado (base reutilizada) | Pipeline de entrenamiento y comparación de modelos (Entrega 2, reutilizado sin cambios en esta entrega) | Andrés Rueda | Commits `d044ef3`, `6dd5cd6` |
| **API — endpoint de metadatos** | *(Pendiente de ejecutar: agrega `GET /api/v1/model-info` a `churn-api` con los hiperparámetros del modelo, según guía enviada)* | Andrés Rueda | *(pendiente — ver `Guia_Andres_Entrega3.zip`)* |
| **Documentación de arquitectura** | *(Pendiente de ejecutar: agrega la sección "Arquitectura (Entrega 3)" al `README.md` describiendo `churn-api` + `churn-tablero`)* | Andrés Rueda | *(pendiente — ver `Guia_Andres_Entrega3.zip`)* |
| **Revisión y retroalimentación** | *(Completar: revisión del reporte final, participación en la sesión de retroalimentación a otro grupo — punto obligatorio de la rúbrica, 5 pts)* | Andrés Rueda | *(pendiente)* |
| **Presentación** | *(Completar: rol de cada integrante en la preparación/exposición de la presentación final de 10 min)* | Ambos | *(pendiente)* |

> **Nota para completar antes de entregar:** las cuatro filas en negrita deben llenarse coordinando con Andrés. Las dos primeras se resuelven siguiendo `Guia_Andres_Entrega3.zip` (dos commits puntuales, no tediosos) — al recibir su captura de commits, reemplazar el hash en estas filas y en la tabla de commits del reporte principal (sección 7). La retroalimentación a otro grupo debe darla **cada integrante** del equipo por separado (rúbrica: 5 pts si ambos participan, penalización si solo uno lo hace).
