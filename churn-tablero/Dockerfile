FROM python:3.12-slim

# Crear usuario que ejecuta la app
RUN adduser --disabled-password --gecos '' tablero-user

WORKDIR /opt/churn-tablero

# Instalar dependencias
COPY ./churn-tablero/requirements.txt /opt/churn-tablero/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /opt/churn-tablero/requirements.txt

# Codigo del tablero
COPY ./churn-tablero/app /opt/churn-tablero/app
COPY ./churn-tablero/run.sh /opt/churn-tablero/run.sh

# Datos limpios (generados por train_final_model.py) que el tablero necesita
# para listar clientes y construir la Matriz de Priorizacion
COPY ./telco_churn_clean.csv /opt/churn-tablero/telco_churn_clean.csv

RUN chmod +x /opt/churn-tablero/run.sh
RUN chown -R tablero-user:tablero-user ./

USER tablero-user
EXPOSE 8501

CMD ["bash", "./run.sh"]
