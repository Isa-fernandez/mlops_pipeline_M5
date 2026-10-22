'''
model_deploy.py
---------------
PIM5 · Avance 4 · API del modelo de riesgo crediticio con FastAPI.

Qué hace este módulo:
    1. Carga el modelo entrenado (models/modelo_final.joblib), que es un paquete con el pipeline
       completo (limpieza + preprocesamiento + LightGBM), el umbral de decisión y la lista de
       las 20 columnas de entrada.
    2. Expone un servicio web con FastAPI para predicciones POR LOTES (batch):
         - POST /predict      : recibe JSON  {"data": [registro, registro, ...]}
         - POST /predict/csv  : recibe un archivo CSV con un registro por fila
       Ambos aplican las mismas transformaciones del entrenamiento (vienen dentro del pipeline)
       y devuelven, por registro, la probabilidad de no pago, la predicción y la alerta.
    3. Endpoints de apoyo: GET /health (estado del servicio) y GET /modelo (información del modelo).

Idea central: la API recibe los datos CRUDOS, tal como llegan del sistema de créditos.
No repite ninguna transformación a mano: todo está dentro del pipeline guardado, así que
lo que se entrenó y lo que se despliega es exactamente lo mismo.

Cómo ejecutarla (desde la raíz del repo, con el entorno virtual activo):
    uvicorn src.model_deploy:app --reload
Documentación interactiva (Swagger): http://localhost:8000/docs
Con Docker: ver Dockerfile en la raíz del repo.
'''

import io
import sys
from pathlib import Path
from typing import List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# 1. RUTAS Y CARGA DEL MODELO
# ---------------------------------------------------------------------------
RUTA_SRC = Path(__file__).resolve().parent
RUTA_MODELO = RUTA_SRC.parent / 'models' / 'modelo_final.joblib'

# El pipeline guardado usa funciones de ft_engineering.py (limpiar_datos, _log_montos).
# Para que joblib pueda reconstruirlo, ese módulo tiene que poder importarse: se agrega src/
# a la ruta de búsqueda de Python. Así la API funciona igual si se lanza desde la raíz
# (uvicorn src.model_deploy:app), desde src/ (python model_deploy.py) o dentro de Docker.
if str(RUTA_SRC) not in sys.path:
    sys.path.insert(0, str(RUTA_SRC))

MAX_REGISTROS = 20_000   # límite por solicitud: evita lotes gigantes que bloqueen el servidor


def cargar_paquete(ruta: Path = RUTA_MODELO) -> dict:
    '''Carga el paquete del modelo. Si no existe, explica cómo generarlo.'''
    if not ruta.exists():
        raise FileNotFoundError(
            f'No existe {ruta}. Generarlo con: python src/model_training_evaluation.py'
        )
    return joblib.load(ruta)


# Se carga UNA sola vez, al iniciar el servidor (no en cada solicitud).
paquete = cargar_paquete()
PIPELINE = paquete['pipeline']
UMBRAL = float(paquete['umbral_no_pago'])
COLUMNAS = list(paquete['columnas_entrada'])
INDICE_NO_PAGO = list(PIPELINE.classes_).index(0)   # la clase de interés es 0 = 'no pagó'


# ---------------------------------------------------------------------------
# 2. ESQUEMAS DE DATOS (Pydantic)
# ---------------------------------------------------------------------------
# Pydantic valida cada registro antes de llegar al modelo: si falta un campo o un valor no es
# un número, la API responde un error 422 claro, en lugar de fallar adentro del modelo.
# Todos los campos aceptan null: el pipeline imputa los faltantes como en el entrenamiento.

class Credito(BaseModel):
    '''Un registro: las 20 variables crudas de una solicitud de crédito.'''
    model_config = ConfigDict(
        json_schema_extra={
            'example': {
                'capital_prestado': 1200000.0, 'salario_cliente': 2500000.0,
                'total_otros_prestamos': 1500000.0, 'cuota_pactada': 120000.0,
                'saldo_mora': 0.0, 'saldo_total': 2000000.0, 'saldo_principal': 1800000.0,
                'saldo_mora_codeudor': 0.0, 'promedio_ingresos_datacredito': 2400000.0,
                'plazo_meses': 10, 'edad_cliente': 35, 'puntaje_datacredito': 780.0,
                'cant_creditosvigentes': 3, 'huella_consulta': 4,
                'creditos_sectorFinanciero': 2, 'creditos_sectorCooperativo': 0,
                'creditos_sectorReal': 1, 'tipo_credito': 4, 'tipo_laboral': 'Empleado',
                'tendencia_ingresos': 'Creciente',
            }
        }
    )

    # Montos
    capital_prestado: Optional[float] = Field(None, description='Monto del crédito')
    salario_cliente: Optional[float] = None
    total_otros_prestamos: Optional[float] = None
    cuota_pactada: Optional[float] = None
    saldo_mora: Optional[float] = None
    saldo_total: Optional[float] = None
    saldo_principal: Optional[float] = None
    saldo_mora_codeudor: Optional[float] = None
    promedio_ingresos_datacredito: Optional[float] = None
    # Numéricas (conteos, edad, puntaje externo)
    plazo_meses: Optional[float] = None
    edad_cliente: Optional[float] = None
    puntaje_datacredito: Optional[float] = None
    cant_creditosvigentes: Optional[float] = None
    huella_consulta: Optional[float] = None
    creditos_sectorFinanciero: Optional[float] = None
    creditos_sectorCooperativo: Optional[float] = None
    creditos_sectorReal: Optional[float] = None
    # Categóricas
    tipo_credito: Optional[Union[int, str]] = Field(
        None, description='Código del producto (4, 9, 10, 6; otros códigos se agrupan en "Otros")')
    tipo_laboral: Optional[str] = Field(None, description='Empleado o Independiente')
    tendencia_ingresos: Optional[str] = Field(None, description='Creciente, Decreciente o Estable')


class LoteCreditos(BaseModel):
    '''Solicitud por lotes: una lista de registros (mismo formato que usa el monitoreo).'''
    data: List[Credito] = Field(..., min_length=1, max_length=MAX_REGISTROS)


class ResultadoCredito(BaseModel):
    '''Resultado de un registro.'''
    prob_no_pago: float = Field(..., description='Probabilidad de NO pagar a tiempo (0 a 1)')
    prediccion_pago_atiempo: int = Field(..., description='1 = pagará a tiempo, 0 = no pagará (mismo código que Pago_atiempo)')
    alerta: bool = Field(..., description='True si la probabilidad de no pago supera el umbral: revisar el crédito')


class RespuestaPrediccion(BaseModel):
    '''Respuesta del endpoint de predicción.'''
    modelo: str
    umbral_no_pago: float
    n_registros: int
    n_alertas: int
    predictions: List[int] = Field(..., description='Predicción de Pago_atiempo por registro (1 = paga, 0 = no paga)')
    resultados: List[ResultadoCredito]


# ---------------------------------------------------------------------------
# 3. LÓGICA DE PREDICCIÓN
# ---------------------------------------------------------------------------
def predecir(df: pd.DataFrame) -> dict:
    '''
    Recibe un DataFrame con las columnas crudas y devuelve la respuesta de la API.
    El pipeline aplica la limpieza por reglas, la imputación y el One-Hot del entrenamiento.
    '''
    faltantes = [c for c in COLUMNAS if c not in df.columns]
    if faltantes:
        raise HTTPException(status_code=422, detail=f'Faltan columnas: {faltantes}')

    X = df[COLUMNAS]   # solo las 20 columnas del modelo, en el orden del entrenamiento
    try:
        prob = PIPELINE.predict_proba(X)[:, INDICE_NO_PAGO]
    except Exception as error:   # p. ej. texto en una columna numérica de un CSV
        raise HTTPException(status_code=422, detail=f'No se pudo predecir: {error}') from error

    alerta = prob >= UMBRAL
    pred_pago = np.where(alerta, 0, 1)   # alerta -> se predice 'no pagará' (Pago_atiempo = 0)
    resultados = [
        {'prob_no_pago': round(float(p), 6), 'prediccion_pago_atiempo': int(y), 'alerta': bool(a)}
        for p, y, a in zip(prob, pred_pago, alerta)
    ]
    return {
        'modelo': paquete['nombre_modelo'],
        'umbral_no_pago': round(UMBRAL, 6),
        'n_registros': len(resultados),
        'n_alertas': int(alerta.sum()),
        'predictions': [int(y) for y in pred_pago],
        'resultados': resultados,
    }


# ---------------------------------------------------------------------------
# 4. APLICACIÓN FASTAPI Y ENDPOINTS
# ---------------------------------------------------------------------------
app = FastAPI(
    title='API de riesgo crediticio (PIM5)',
    description=(
        'Predice la probabilidad de que un crédito **no se pague a tiempo**. '
        'Acepta lotes de registros en JSON (`/predict`) o en un archivo CSV (`/predict/csv`). '
        f'La alerta se activa cuando la probabilidad de no pago es ≥ {UMBRAL:.3f}. '
        'El modelo tiene capacidad de discriminación limitada (ROC-AUC ≈ 0,68): sirve para '
        'priorizar revisiones, no para decidir solo.'
    ),
    version='1.0.0',
)


@app.get('/', include_in_schema=False)
def inicio():
    '''La raíz redirige a la documentación interactiva.'''
    return RedirectResponse(url='/docs')


@app.get('/health', tags=['Estado'])
def health():
    '''Comprueba que el servicio está en línea y el modelo cargado.'''
    return {'status': 'ok', 'modelo_cargado': True, 'modelo': paquete['nombre_modelo']}


@app.get('/modelo', tags=['Estado'])
def info_modelo():
    '''Información del modelo desplegado: nombre, umbral, columnas esperadas y métricas en test.'''
    metricas = {k: round(float(v), 4) for k, v in paquete.get('metricas_test', {}).items()}
    return {
        'modelo': paquete['nombre_modelo'],
        'umbral_no_pago': round(UMBRAL, 6),
        'columnas_entrada': COLUMNAS,
        'metricas_test': metricas,
        'max_registros_por_solicitud': MAX_REGISTROS,
    }


@app.post('/predict', response_model=RespuestaPrediccion, tags=['Predicción'])
def predict(lote: LoteCreditos):
    '''
    Predicción por lotes en **JSON**: `{"data": [registro, registro, ...]}`.
    Cada registro lleva las 20 variables crudas; los valores desconocidos se envían como `null`.
    '''
    df = pd.DataFrame([registro.model_dump() for registro in lote.data])
    return predecir(df)


@app.post('/predict/csv', response_model=RespuestaPrediccion, tags=['Predicción'])
async def predict_csv(archivo: UploadFile = File(..., description='CSV con un registro por fila')):
    '''
    Predicción por lotes desde un archivo **CSV** (separado por comas, con encabezado).
    Debe incluir las 20 columnas del modelo; las columnas de más (fecha, Pago_atiempo, etc.)
    se ignoran. Las celdas vacías se tratan como dato faltante.
    '''
    if not (archivo.filename or '').lower().endswith('.csv'):
        raise HTTPException(status_code=415, detail='El archivo debe tener extensión .csv')
    contenido = await archivo.read()
    try:
        df = pd.read_csv(io.BytesIO(contenido))
    except Exception as error:
        raise HTTPException(status_code=400, detail=f'No se pudo leer el CSV: {error}') from error
    if df.empty:
        raise HTTPException(status_code=400, detail='El CSV no tiene filas.')
    if len(df) > MAX_REGISTROS:
        raise HTTPException(status_code=413, detail=f'Máximo {MAX_REGISTROS} filas por archivo.')
    return predecir(df)


# Permite ejecutar el archivo directamente para pruebas locales (desde src/):
#     python model_deploy.py
if __name__ == '__main__':
    import uvicorn
    uvicorn.run('model_deploy:app', host='0.0.0.0', port=8000, reload=True)
