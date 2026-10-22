'''
ft_engineering.py
-----------------
PIM5 · Avance 2 (V1.1.0) · Ingeniería de características del modelo de riesgo crediticio.

Qué hace este módulo:
    1. Limpia los datos con reglas fijas definidas en el EDA (valores imposibles o inválidos).
    2. Define el preprocesador: imputación, indicadores de dato faltante, transformación
       logarítmica, escalado y codificación One-Hot.
    3. Divide la base en entrenamiento y prueba (estratificada o temporal).

Idea central: el preprocesador NUNCA se ajusta (fit) con toda la base, solo con entrenamiento.
Por eso este módulo no devuelve datos transformados: devuelve un *pipeline* sin ajustar que
se ajusta dentro de cada entrenamiento (y dentro de cada fold de la validación cruzada).
Así no hay fuga de información (data leakage) de la partición de prueba hacia el modelo.

Uso típico:
    from ft_engineering import ft_engineering
    datos = ft_engineering(con_puntaje=False)
    X_train, X_test = datos['X_train'], datos['X_test']
'''

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# 1. CONSTANTES
# ---------------------------------------------------------------------------
SEMILLA = 42             # misma semilla en todo el proyecto (reproducibilidad)
TAMANO_TEST = 0.20       # 80 % entrenamiento / 20 % prueba
TARGET = 'Pago_atiempo'  # 1 = pagó a tiempo, 0 = no pagó (la clase de interés es la 0)
COLUMNA_FECHA = 'fecha_prestamo'

# Reglas de limpieza (decididas en el EDA, secciones 5 y 11.3)
EDAD_MAXIMA_PLAUSIBLE = 90                                  # edades > 90 son un valor centinela (121-123)
CATEGORIAS_TENDENCIA = ['Creciente', 'Decreciente', 'Estable']
CATEGORIAS_TIPO_CREDITO = ['4', '9', '10', '6']             # el resto (7, 68 y códigos nuevos) pasa a 'Otros'
ETIQUETA_SIN_DATO = 'Sin dato'

# Variables numéricas con montos: distribución muy asimétrica -> se les aplica log(1 + x)
COLUMNAS_MONTOS = [
    'capital_prestado',
    'salario_cliente',
    'total_otros_prestamos',
    'cuota_pactada',
    'saldo_mora',
    'saldo_total',
    'saldo_principal',
    'saldo_mora_codeudor',
    'promedio_ingresos_datacredito',
]

# Resto de las variables numéricas (conteos, edad, puntaje externo)
COLUMNAS_NUMERICAS = [
    'plazo_meses',
    'edad_cliente',
    'puntaje_datacredito',
    'cant_creditosvigentes',
    'huella_consulta',
    'creditos_sectorFinanciero',
    'creditos_sectorCooperativo',
    'creditos_sectorReal',
]

# Categóricas nominales (ninguna tiene orden): One-Hot
COLUMNAS_CATEGORICAS = ['tipo_credito', 'tipo_laboral', 'tendencia_ingresos']

# Variable con fuga de información (EDA, sección 9): solo se usa para documentar su efecto
COLUMNA_PUNTAJE = 'puntaje'

# Redundante con saldo_total (Spearman 0,95): se descarta SOLO en el modelo lineal.
# Los modelos de árboles toleran la redundancia, por eso allí se conserva.
COLUMNAS_REDUNDANTES_LINEAL = ['saldo_principal']


# ---------------------------------------------------------------------------
# 2. LIMPIEZA POR REGLAS
# ---------------------------------------------------------------------------
def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Aplica las reglas de limpieza del EDA y devuelve una copia (no modifica el original).

    Son reglas FIJAS (no aprenden nada de los datos), por lo que se pueden aplicar a toda la
    base sin riesgo de fuga de información. Las reglas que sí aprenden de los datos
    (medianas, escalado, categorías) van en el preprocesador y se ajustan solo con train.
    '''
    df = df.copy()

    # tendencia_ingresos: solo son válidas 3 categorías; lo demás (p. ej. números) pasa a nulo
    if 'tendencia_ingresos' in df.columns:
        df['tendencia_ingresos'] = df['tendencia_ingresos'].where(
            df['tendencia_ingresos'].isin(CATEGORIAS_TENDENCIA), np.nan
        )

    # edad_cliente: edades imposibles (> 90) pasan a nulo y luego se imputan
    if 'edad_cliente' in df.columns:
        df['edad_cliente'] = df['edad_cliente'].where(
            df['edad_cliente'] <= EDAD_MAXIMA_PLAUSIBLE, np.nan
        )

    # puntaje_datacredito: 0 y negativos son códigos de 'sin dato' -> nulo
    if 'puntaje_datacredito' in df.columns:
        df['puntaje_datacredito'] = df['puntaje_datacredito'].where(
            df['puntaje_datacredito'] > 0, np.nan
        )

    # tipo_credito es un código de producto (nominal): se pasa a texto y se agrupan los raros.
    # Cualquier código fuera de la lista (incluidos los que aparezcan en producción) -> 'Otros'
    if 'tipo_credito' in df.columns:
        # to_numeric acepta tanto 4 como '4' (la API puede enviar texto); lo no numérico -> 'Otros'
        codigos = pd.to_numeric(df['tipo_credito'], errors='coerce').astype('Int64').astype(str)
        df['tipo_credito'] = codigos.where(codigos.isin(CATEGORIAS_TIPO_CREDITO), 'Otros')

    return df


def _log_montos(x):
    '''log(1 + x) con piso en 0: si llegara un monto negativo en producción, no genera NaN.'''
    return np.log1p(np.clip(x, 0, None))


# ---------------------------------------------------------------------------
# 3. PREPROCESADOR
# ---------------------------------------------------------------------------
def columnas_entrada(con_puntaje: bool = False, escalar: bool = False) -> list:
    '''
    Lista de columnas CRUDAS que necesita el pipeline como entrada.
    Es también el esquema que usará la API (Avance 4).
    '''
    columnas = COLUMNAS_MONTOS + COLUMNAS_NUMERICAS + COLUMNAS_CATEGORICAS
    if con_puntaje:
        columnas = columnas + [COLUMNA_PUNTAJE]
    if escalar:
        columnas = [c for c in columnas if c not in COLUMNAS_REDUNDANTES_LINEAL]
    return columnas


def crear_preprocesador(escalar: bool = False, con_puntaje: bool = False) -> ColumnTransformer:
    '''
    Construye el ColumnTransformer (sin ajustar).

    Parámetros
    ----------
    escalar : True para modelos sensibles a la escala (regresión logística). Aplica log a los
              montos, estandariza y descarta la variable redundante. False para árboles.
    con_puntaje : True solo para el experimento que documenta la fuga de información.
    '''
    montos = [c for c in COLUMNAS_MONTOS if not (escalar and c in COLUMNAS_REDUNDANTES_LINEAL)]
    numericas = list(COLUMNAS_NUMERICAS)
    if con_puntaje:
        numericas.append(COLUMNA_PUNTAJE)

    # Montos: mediana + indicador de faltante (+ log + escalado si el modelo lo necesita).
    # add_indicator=True agrega una columna 0/1 'faltaba el dato' por cada variable con nulos
    # en train: la ausencia de ingresos puede aportar información por sí misma (EDA, 5.2).
    pasos_montos = [('imputar', SimpleImputer(strategy='median', add_indicator=True))]
    if escalar:
        pasos_montos.append(('log', FunctionTransformer(_log_montos, feature_names_out='one-to-one')))
        pasos_montos.append(('escalar', StandardScaler()))

    pasos_numericas = [('imputar', SimpleImputer(strategy='median', add_indicator=True))]
    if escalar:
        pasos_numericas.append(('escalar', StandardScaler()))

    # Categóricas: los nulos pasan a la categoría 'Sin dato' y luego One-Hot.
    # handle_unknown='ignore': una categoría nueva en producción no rompe la predicción.
    pasos_categoricas = [
        ('imputar', SimpleImputer(strategy='constant', fill_value=ETIQUETA_SIN_DATO)),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ]

    preprocesador = ColumnTransformer(
        transformers=[
            ('montos', Pipeline(pasos_montos), montos),
            ('numericas', Pipeline(pasos_numericas), numericas),
            ('categoricas', Pipeline(pasos_categoricas), COLUMNAS_CATEGORICAS),
        ],
        remainder='drop',                  # cualquier otra columna (p. ej. la fecha) se descarta
        verbose_feature_names_out=False,   # nombres de salida legibles
    )
    return preprocesador


def crear_pipeline_features(escalar: bool = False, con_puntaje: bool = False) -> Pipeline:
    '''
    Pipeline completo de características: limpieza por reglas + preprocesador.
    Recibe el DataFrame CRUDO y entrega un DataFrame numérico sin nulos.
    Al incluir la limpieza dentro del pipeline, la API de producción puede recibir los datos
    tal cual llegan, sin repetir esas reglas en otro lugar.
    '''
    return Pipeline([
        ('limpieza', FunctionTransformer(limpiar_datos)),
        ('preprocesador', crear_preprocesador(escalar=escalar, con_puntaje=con_puntaje)),
    ]).set_output(transform='pandas')


# ---------------------------------------------------------------------------
# 4. PARTICIÓN TRAIN / TEST
# ---------------------------------------------------------------------------
def dividir_datos(df: pd.DataFrame, con_puntaje: bool = False, temporal: bool = False) -> dict:
    '''
    Divide la base en entrenamiento y prueba.

    temporal=False (por defecto): aleatoria ESTRATIFICADA por Pago_atiempo (semilla 42), que
        mantiene la proporción 95/5 en ambos conjuntos.
    temporal=True: el 80 % más antiguo para entrenar y el 20 % más reciente para probar.
        Simula el uso real (entrenar con el pasado, predecir el futuro) y sirve como control
        porque la cartera cambió con el tiempo (EDA, sección 10).
    '''
    columnas = columnas_entrada(con_puntaje=con_puntaje)
    y = df[TARGET]

    if temporal:
        orden = df[COLUMNA_FECHA].sort_values().index
        corte = int(len(orden) * (1 - TAMANO_TEST))
        idx_train, idx_test = orden[:corte], orden[corte:]
        return {
            'X_train': df.loc[idx_train, columnas], 'X_test': df.loc[idx_test, columnas],
            'y_train': y.loc[idx_train], 'y_test': y.loc[idx_test],
        }

    X_train, X_test, y_train, y_test = train_test_split(
        df[columnas], y, test_size=TAMANO_TEST, stratify=y, random_state=SEMILLA
    )
    return {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}


def ft_engineering(df: pd.DataFrame = None, con_puntaje: bool = False, temporal: bool = False) -> dict:
    '''
    Punto de entrada del módulo. Devuelve un diccionario con:
        X_train, X_test, y_train, y_test : partición con las columnas crudas necesarias
        pipeline_arbol  : pipeline de características para modelos de árboles (sin escalar)
        pipeline_lineal : pipeline de características para modelos lineales (log + escalado)
    Los pipelines vienen SIN ajustar: se ajustan al entrenar cada modelo.
    '''
    if df is None:
        # Importación diferida: así el módulo se puede importar sin tener el archivo de carga
        from cargar_datos import cargarDatos
        df = cargarDatos()

    datos = dividir_datos(df, con_puntaje=con_puntaje, temporal=temporal)
    datos['pipeline_arbol'] = crear_pipeline_features(escalar=False, con_puntaje=con_puntaje)
    datos['pipeline_lineal'] = crear_pipeline_features(escalar=True, con_puntaje=con_puntaje)
    return datos


if __name__ == '__main__':
    # Prueba rápida desde la terminal: python ft_engineering.py
    datos = ft_engineering()
    pipe = datos['pipeline_arbol'].fit(datos['X_train'])
    X_tr = pipe.transform(datos['X_train'])
    print(f"Train: {X_tr.shape[0]:,} filas x {X_tr.shape[1]} características")
    print(f"Nulos tras el preprocesado: {int(X_tr.isna().sum().sum())}")
