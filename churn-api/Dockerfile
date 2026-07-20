FROM python:3.12-slim

# Crear usuario que ejecuta la app
RUN adduser --disabled-password --gecos '' api-user

# Instalar compiladores C/C++ y build tools (requerido por scikit-learn)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/churn-api

# Instalar dependencias
COPY ./churn-api/requirements.txt /opt/churn-api/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /opt/churn-api/requirements.txt

# Codigo de la API
COPY ./churn-api/app /opt/churn-api/app
COPY ./churn-api/run.sh /opt/churn-api/run.sh

# Modelo empaquetado (generado por train_final_model.py)
COPY ./modelo_churn_final.joblib /opt/churn-api/modelo_churn_final.joblib

RUN chmod +x /opt/churn-api/run.sh
RUN chown -R api-user:api-user ./

USER api-user
EXPOSE 8001

CMD ["bash", "./run.sh"]
