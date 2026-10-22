# PIM5 — Despliegue de un modelo de riesgo crediticio

**Soy Henry · Bootcamp de Data Science · Módulo 5: Fundamentos de nube y ciencia de datos de producción**
**Autora:** María Isabel Fernández
**Repositorio:** https://github.com/Isa-fernandez/mlops_pipeline_M5

---

## 1. Caso de negocio

Una empresa financiera necesita anticipar el comportamiento de sus nuevos clientes a partir de información histórica de créditos. El modelo predice si un cliente **pagará a tiempo** (`Pago_atiempo`: 1 = pagó a tiempo, 0 = no pagó) para apoyar la decisión de otorgar el crédito.

El foco del proyecto es la **puesta en producción**: repositorio versionado, pipeline reproducible, API, imagen Docker, monitoreo de *data drift* y una aplicación en Streamlit. La performance del modelo es secundaria: la base es sintética y de baja calidad (ver sección 9).

## 2. Estructura del repositorio

```
mlops_pipeline_M5/                  (raíz del repo)
├── src/
│   ├── cargar_datos.py             carga de la base (función cargarDatos)
│   ├── comprension_eda.ipynb       EDA: inicial, univariable, bivariable, multivariable
│   ├── ft_engineering.py           limpieza e ingeniería de características
│   ├── model_training_evaluation.py     entrenamiento, evaluación y selección
│   ├── model_training_evaluation.ipynb  notebook con el análisis de los modelos
│   ├── model_deploy.py             API con FastAPI
│   ├── model_monitoring.py         monitoreo y detección de data drift
│   └── app_streamlit.py            tablero de monitoreo y simulador
├── models/
│   └── modelo_final.joblib         modelo entrenado (pipeline completo)
├── Base_de_datos.csv
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── .gitignore
└── README.md
```

**Notas sobre la estructura:**
- La carga de datos se implementó como script (`cargar_datos.py`) y no como notebook.
- Se agregaron archivos nuevos necesarios (`app_streamlit.py`, `Dockerfile`, `.dockerignore`, `models/`) sin modificar los directorios ni los nombres de la estructura base, tal como lo confirmó el equipo docente.
- Los datos se leen de un archivo dentro del repositorio porque así lo exige el trabajo; en un entorno real se cargarían desde un data warehouse o similar.

## 3. Flujo de trabajo en Git

- **Ramas:** `developer`, `certification` y `main` (rama principal, equivalente a `master`).
- **Flujo:** commits en `developer` → merge a `certification` → Pull Request de `certification` a `main` → sincronización de las tres ramas.
- **Mensajes de commit** convencionales (`feat:`, `chore:`, `docs:`).

| Versión | Contenido |
|---|---|
| V1.0.0 | Estructura inicial (`.gitignore`, `requirements.txt`, base de datos, `src/`) |
| V1.0.1 | Carga de datos y EDA (Avance 1) |
| V1.1.0 | Ingeniería de características (Avance 2) |
| V1.1.1 | Entrenamiento, evaluación y modelo final (Avance 2) |
| V1.2.0 | Monitoreo de data drift y app Streamlit (Avance 3) |
| V1.3.0 | API FastAPI, imagen Docker y Streamlit conectado a la API (Avance 4) |
| V1.3.1 | README final |

## 4. Datos

- 10.763 filas × 23 columnas, sin duplicados. Período de `fecha_prestamo`: 26/11/2024 a 26/04/2026.
- **Target:** `Pago_atiempo`. Fuerte desbalance: 95,25 % pagó a tiempo y 4,75 % no pagó.
- **Nulos:** `promedio_ingresos_datacredito` (2.930), `tendencia_ingresos` (2.932), `saldo_mora_codeudor` (590), `saldo_principal` (405), `saldo_total` (156), `saldo_mora` (156) y `puntaje_datacredito` (6).
- `tendencia_ingresos` contiene 58 valores numéricos inválidos (se tratan como nulos).
- `tipo_credito` tiene categorías con muy pocos casos (7 y 68).

## 5. Hallazgos del EDA (Avance 1)

- **Fuga de información en `puntaje`:** correlación de 0,92 con el target y rangos casi disjuntos (no pagaron: máximo 62,7; pagaron: mínimo 63,8). Con esa variable cualquier modelo alcanza ROC-AUC = 1,0. Dado que la base es sintética, es probable que sea un artefacto de su generación. **Se excluye del modelo.**
- Sin `puntaje`, la señal es débil: el mejor AUC individual es 0,623 (`puntaje_datacredito`).
- 150 edades imposibles (121 a 123 años) y 145 `puntaje_datacredito` iguales a 0.
- Salarios y préstamos con colas extremas (hasta 22.000 millones).
- Redundancia entre `saldo_total` y `saldo_principal` (Spearman 0,95) y multicolinealidad en `cant_creditosvigentes` (VIF 10,05).
- La cartera cambió en el tiempo: plazos más cortos, más consultas, cuotas y saldos mayores.

## 6. Ingeniería de características y modelado (Avance 2)

**Limpieza por reglas:** `tendencia_ingresos` inválida y edad > 90 pasan a nulo; `puntaje_datacredito` ≤ 0 pasa a nulo; `tipo_credito` fuera de 4, 9, 10 y 6 se agrupa como "Otros".

**Preprocesamiento (dentro de un `Pipeline` de scikit-learn, ajustado solo con datos de entrenamiento):**
- Numéricas: imputación con mediana + indicador de faltante.
- Categóricas: categoría "Sin dato" + One-Hot.
- Solo para el modelo lineal: logaritmo de montos y escalado.

**Modelado:** la clase de interés es la 0 (no pagó), por lo que las métricas (Precision, Recall, F1, PR-AUC) se calculan sobre esa clase. Desbalanceo tratado con `class_weight='balanced'` y `scale_pos_weight` (sin SMOTE). Partición estratificada 80/20 (semilla 42) y validación cruzada de 5 folds.

| Modelo | ROC-AUC CV | PR-AUC CV | ROC-AUC test | PR-AUC test |
|---|---|---|---|---|
| Referencia | 0,500 | 0,048 | 0,500 | 0,047 |
| Regresión Logística | 0,674 ± 0,028 | 0,134 ± 0,020 | 0,693 | 0,133 |
| Árbol de Decisión | 0,639 ± 0,017 | 0,092 ± 0,015 | 0,616 | 0,069 |
| Random Forest | 0,679 ± 0,026 | 0,148 ± 0,013 | 0,682 | 0,129 |
| XGBoost | 0,686 ± 0,016 | 0,146 ± 0,019 | 0,688 | 0,142 |
| **LightGBM (elegido)** | **0,686 ± 0,022** | **0,152 ± 0,020** | **0,684** | **0,159** |

- **Selección:** mayor PR-AUC medio en validación cruzada (solo entrenamiento), sin modelos que usen `puntaje`. Cuatro modelos empatan técnicamente (diferencias menores al desvío).
- **Umbral de decisión:** 0,601, el que maximiza el F1 de la clase "no pagó" con predicciones out-of-fold. En test se revisa el 12 % de los créditos y se detecta el 28 % de los no pagos (Precision 0,111; F1 0,160). El 10 % de mayor riesgo concentra el 26,5 % de los no pagos.
- **Intervalos de confianza (bootstrap 95 %, test):** ROC-AUC [0,620 ; 0,737]; PR-AUC [0,101 ; 0,229].
- **Control temporal:** ROC-AUC entre 0,63 y 0,66 con partición por fecha.

## 7. Monitoreo y data drift (Avance 3)

`src/model_monitoring.py` compara una **referencia** (80 % más antiguo por `fecha_prestamo`: 8.610 créditos, 26/11/2024 a 12/07/2025) contra la cartera **actual** (20 % más reciente: 2.153 créditos, 12/07/2025 a 26/04/2026).

**Métricas:** PSI (criterio del semáforo: < 0,10 estable, 0,10 a 0,25 moderado, ≥ 0,25 alto), divergencia de Jensen-Shannon, KS (numéricas) y chi-cuadrado (categóricas). El p-valor no decide, porque con miles de filas todo resulta "significativo".

**Resultados:**
- 0 variables con PSI alto y 4 con PSI moderado: `promedio_ingresos_datacredito` (0,235), `total_otros_prestamos` (0,222), `plazo_meses` (0,109) y `salario_cliente` (0,107). El resto es estable.
- PSI del score: 0,048 (estable), pero la tasa de alertas sube de 11,7 % a 14,5 % (+24 % relativo).
- Evolución trimestral del PSI del score (ajustado por ruido de muestra): 0,045 → 0,067 → 0,073. Tendencia creciente moderada, sin cambios bruscos.
- **Recomendación automática:** mantener el modelo, vigilarlo y preparar el reentrenamiento si el PSI del score se acerca a 0,25.
- **Control del método:** train vs. test aleatorio da PSI máximo 0,010 (el método no inventa drift).

**Alertas automáticas:** niveles Crítica, Advertencia y Sin alertas, cada una con acción sugerida. Los umbrales de PSI son el estándar; los de salto y tendencia son heurísticos y configurables.

**Aplicación Streamlit** (`src/app_streamlit.py`), con pestañas de estado general (semáforo y tabla de métricas), detalle por variable, evolución temporal, alertas y recomendaciones, datos y pronósticos, y simulador de un cliente nuevo.

```bash
streamlit run src/app_streamlit.py
```

Como alternativa con librería para el monitoreo se podría usar Evidently; se descartó por riesgo de incompatibilidad de versiones.

## 8. API y Docker (Avance 4)

`src/model_deploy.py` carga `models/modelo_final.joblib` una sola vez al arrancar y recibe datos **crudos** (20 columnas); las transformaciones están dentro del pipeline.

| Endpoint | Descripción |
|---|---|
| `GET /health` | Estado del servicio |
| `GET /modelo` | Nombre del modelo, umbral, columnas y métricas en test |
| `POST /predict` | Predicción por lotes en JSON: `{"data": [...]}` |
| `POST /predict/csv` | Predicción por lotes subiendo un archivo CSV |

La respuesta incluye `predictions` (1 = paga, 0 = no paga) y, por registro, `prob_no_pago`, `prediccion_pago_atiempo` y `alerta` (probabilidad ≥ 0,601). Se valida la entrada con Pydantic (error 422 ante campos inválidos, 415 si el archivo no es CSV, límite de 20.000 registros por solicitud). Documentación automática en `/docs`.

**Ejecución local:**

```bash
uvicorn src.model_deploy:app --port 8000
```

**Docker:**

```bash
docker build -t pim5-api .
docker run -d -p 8000:8000 --name pim5-api pim5-api
```

La imagen usa `python:3.13-slim` (misma versión que el modelo), instala `libgomp1` (requerida por LightGBM), copia solo `src/` y `models/`, corre con un usuario sin privilegios y tiene un `HEALTHCHECK` sobre `/health`. Las dependencias salen del `requirements.txt` completo. En producción real se preferiría una imagen mínima con solo las librerías de la API (menor tamaño, arranque más rápido y menos superficie de ataque).

Verificación: la API devuelve las mismas probabilidades que el modelo local (diferencia < 0,000001) y el simulador de Streamlit, con la API en línea, muestra "Calculado con: API".

## 9. Limitaciones

- Base sintética y de baja calidad: las métricas pueden dar resultados atípicos y se documentan sin forzarlas.
- Posible fuga de información en `puntaje` (excluida del modelo).
- Fuerte desbalance (95/5) y solo 102 no pagos en el conjunto de prueba: métricas inestables.
- Cuatro modelos casi empatados.
- Muchas falsas alarmas (Precision entre 0,08 y 0,11); el umbral definitivo requiere una matriz de costos real del negocio.
- La cantidad de créditos cae con el tiempo (1.917 en enero de 2025 contra 11 en abril de 2026), lo que vuelve inestables los trimestres finales del monitoreo.
- El muestreo periódico se hace por ventanas de fecha sobre la base; en producción saldría del registro de predicciones de la API, y la tasa real de no pago llegaría con retraso.

## 10. Cómo reproducir

```bash
git clone https://github.com/Isa-fernandez/mlops_pipeline_M5.git
cd mlops_pipeline_M5
python -m venv venv
source venv/Scripts/activate        # en Windows con Git Bash
pip install -r requirements.txt

python src/model_training_evaluation.py   # entrena y guarda models/modelo_final.joblib
python src/model_monitoring.py            # reporte de drift (genera reports/)
uvicorn src.model_deploy:app --port 8000  # API (o usar Docker)
streamlit run src/app_streamlit.py        # tablero de monitoreo
```

Entorno probado: Python 3.13. Las versiones de las librerías del modelo y la API están fijadas en `requirements.txt`, porque el modelo guardado solo carga bien con las mismas versiones con las que se entrenó.
