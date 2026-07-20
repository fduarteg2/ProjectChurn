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