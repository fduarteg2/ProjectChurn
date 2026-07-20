# Manual de Usuario — Panel de Retención de Clientes (TelcoChurn)

## 1. Acceso al tablero

Abra en su navegador la URL del tablero desplegado:

```
http://<IP_PUBLICA_EC2>:8501
```

Ejemplo (instancia activa de la Entrega 3): `http://3.90.218.62:8501`

El tablero requiere que la API `churn-api` esté corriendo (se despliega junto con el tablero en el mismo `docker compose`). Si al abrir el tablero aparece un mensaje de error indicando que no se pudo conectar con la API, verifique que el contenedor `churn-api` esté activo (ver Manual de Instalación).

## 2. Estructura general

El tablero está organizado en 3 pestañas, ubicadas en la parte superior:

- 🎯 **Simulador de Escenarios**
- 🗺️ **Matriz de Priorización**
- 🔍 **Factores de Influencia**

Todas las predicciones que se muestran en el tablero son calculadas en tiempo real por la API `churn-api` (el tablero no tiene el modelo embebido, solo lo consulta).

## 3. Simulador de Escenarios

Permite explorar cómo cambia la probabilidad de fuga (churn) de un cliente si se modifican ciertas variables de negocio.

**Pasos de uso:**

1. En **"Cliente base (customerID)"**, seleccione un cliente existente de la base de datos. Se cargará su perfil original.
2. Ajuste las variables disponibles:
   - **Tipo de Contrato**: Month-to-month, One year, Two year.
   - **Soporte Técnico**: Yes, No, No internet service.
   - **Servicio de Internet**: DSL, Fiber optic, No.
   - **Método de Pago**: Electronic check, Mailed check, Bank transfer, Credit card.
3. El panel derecho muestra de inmediato:
   - La **probabilidad de fuga** del escenario simulado (0-100%), con una etiqueta de riesgo (ALTO RIESGO si ≥ 50%, BAJO RIESGO si menor).
   - El **cambio (delta)** respecto a la probabilidad del perfil original del cliente.
   - Una tabla con el **perfil simulado** completo.

**Uso recomendado:** un Account Manager puede usar esta vista para evaluar, antes de contactar a un cliente, qué acción de retención (p. ej. ofrecer un contrato anual, o soporte técnico) tendría mayor impacto en reducir su riesgo de fuga.

## 4. Matriz de Priorización

Cruza la **probabilidad de fuga** (eje X) contra la **facturación mensual** (eje Y, proxy del valor del cliente) para identificar qué cuentas priorizar.

**Controles:**

- **Umbral de Alto Riesgo**: define a partir de qué probabilidad se considera un cliente de "alto riesgo" (por defecto 50%).
- **Umbral de Alto Valor**: define a partir de qué facturación mensual se considera un cliente de "alto valor" (por defecto la mediana).

**Interpretación del gráfico:**

- Cada punto es un cliente. El color indica el cuadrante:
  - 🔴 **Zona Crítica** (alto riesgo + alto valor): prioridad máxima de retención.
  - Alto riesgo, bajo valor.
  - Bajo riesgo, alto valor.
  - Bajo riesgo, bajo valor.
- Debajo del gráfico se muestra el conteo de clientes en Zona Crítica sobre la muestra visualizada.
- Puede pasar el cursor sobre cualquier punto para ver el `customerID`, tipo de contrato y antigüedad (tenure) de ese cliente.

**Uso recomendado:** el equipo de gestión de cuentas revisa primero los clientes en Zona Crítica antes de continuar con el resto.

## 5. Factores de Influencia

Muestra qué variables de negocio tienen mayor peso en las predicciones del modelo (importancia nativa del Random Forest, agregada por variable original).

**Controles:**

- **Número de variables a mostrar**: ajusta cuántas barras se listan (entre 5 y 15).

**Interpretación:** las variables con barras más largas son las que más influyen en que el modelo prediga fuga. Esto ayuda al equipo comercial a entender *por qué* un cliente está en riesgo, no solo *que* lo está.

## 6. Preguntas frecuentes

**¿Por qué el tablero muestra un error de conexión?**
La API `churn-api` no está corriendo o no es alcanzable desde el tablero. Ver el Manual de Instalación para verificar el estado de los contenedores.

**¿Los datos de los clientes se actualizan automáticamente?**
No. El tablero usa el archivo `telco_churn_clean.csv` generado en el último entrenamiento (`train_final_model.py`). Para reflejar clientes nuevos, es necesario regenerar ese archivo y reconstruir la imagen del tablero.
