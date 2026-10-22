'''
model_monitoring.py
-------------------
PIM5 · Avance 3 · Monitoreo del modelo de riesgo crediticio y detección de data drift.

Qué es el data drift:
    El modelo se entrenó con datos del pasado. Si los clientes que llegan hoy son distintos
    (otros salarios, otros plazos, otros tipos de crédito), el modelo se usa fuera de la zona
    donde aprendió y sus predicciones pierden confiabilidad. Monitorear es comparar, de forma
    periódica, los datos nuevos contra los de referencia.

Qué hace este módulo:
    1. Compara cada variable de entrada entre un período de REFERENCIA y uno ACTUAL.
    2. Mide el cambio con el PSI (Population Stability Index, el estándar en riesgo crediticio) y
       con la divergencia de Jensen-Shannon, y los complementa con un test estadístico (KS para
       numéricas, chi-cuadrado para categóricas).
    3. Mide el drift de las predicciones del modelo (distribución del score y tasa de alertas).
    4. Calcula la evolución por período (trimestre o mes) y detecta tendencias y cambios bruscos.
    5. Genera alertas y recomendaciones automáticas (reentrenar, revisar variables).
    6. Incluye un control (train vs test aleatorio) que debe dar "sin drift": valida el método.

Lectura del PSI (convención de la industria):
    PSI < 0,10          -> estable
    0,10 <= PSI < 0,25  -> cambio moderado (vigilar)
    PSI >= 0,25         -> cambio alto (revisar / considerar reentrenar)

Nota sobre los tests estadísticos: con muchas filas, el KS y el chi-cuadrado detectan diferencias
mínimas sin importancia práctica. Por eso el semáforo se decide con el PSI y el p-valor es
solo un dato de apoyo.

Uso desde la terminal (desde la carpeta src/):
    python model_monitoring.py
La interfaz gráfica está en app_streamlit.py.
'''

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

from cargar_datos import cargarDatos
from ft_engineering import (
    COLUMNA_FECHA, COLUMNAS_CATEGORICAS, TARGET, columnas_entrada, dividir_datos, limpiar_datos,
)

# ---------------------------------------------------------------------------
# 1. CONSTANTES
# ---------------------------------------------------------------------------
RUTA_SRC = Path(__file__).resolve().parent
RUTA_MODELO = RUTA_SRC.parent / 'models' / 'modelo_final.joblib'
RUTA_REPORTES = RUTA_SRC.parent / 'reports'

PSI_MODERADO = 0.10
PSI_ALTO = 0.25
N_BINS = 10                 # deciles de la referencia para variables numéricas continuas
MAX_UNICOS_DISCRETA = 10    # numéricas con <= 10 valores distintos se tratan como categorías
PROP_REFERENCIA = 0.80      # 80 % más antiguo = referencia; 20 % más reciente = actual
ALFA = 0.01
EPSILON = 1e-4              # evita log(0) cuando una categoría desaparece o aparece
MIN_FILAS_PERIODO = 100     # un período con menos filas no se interpreta (PSI inestable)

# Reglas de alerta (heurísticas configurables, no son estándares: ajustarlas con el negocio)
SALTO_ABRUPTO = 0.05            # aumento del PSI del score entre dos períodos seguidos = cambio brusco
MIN_AUMENTO_TENDENCIA = 0.02    # aumento total mínimo del PSI para hablar de tendencia
PERIODOS_TENDENCIA = 4          # cuántos períodos confiables se miran para la tendencia
CAMBIO_RELATIVO_ALERTAS = 0.20  # variación relativa de la tasa de alertas que se avisa


# ---------------------------------------------------------------------------
# 2. MODELO Y DATOS
# ---------------------------------------------------------------------------
def cargar_modelo(ruta: Path = RUTA_MODELO) -> dict:
    '''Carga el paquete guardado por model_training_evaluation.py (pipeline + umbral + metadata).'''
    if not Path(ruta).exists():
        raise FileNotFoundError(
            f'No existe {ruta}. Generarlo con: python model_training_evaluation.py'
        )
    return joblib.load(ruta)


def proba_no_pago(paquete: dict, X: pd.DataFrame) -> np.ndarray:
    '''Probabilidad de la clase 0 (no pagó), buscada por classes_ y no por posición.'''
    pipe = paquete['pipeline']
    indice = list(pipe.classes_).index(0)
    return pipe.predict_proba(X[paquete['columnas_entrada']])[:, indice]


def cargar_base() -> pd.DataFrame:
    '''Base completa ordenada por fecha, con las reglas de limpieza del EDA ya aplicadas.'''
    df = cargarDatos().sort_values(COLUMNA_FECHA).reset_index(drop=True)
    return limpiar_datos(df)


def separar_periodos(df: pd.DataFrame, prop_referencia: float = PROP_REFERENCIA) -> tuple:
    '''
    Simula el uso real: el modelo conoce el pasado (referencia) y recibe el futuro (actual).
    Devuelve (referencia, actual) cortando la base por fecha.
    '''
    df = df.sort_values(COLUMNA_FECHA)
    corte = int(len(df) * prop_referencia)
    return df.iloc[:corte], df.iloc[corte:]


# ---------------------------------------------------------------------------
# 3. PSI Y TESTS ESTADÍSTICOS
# ---------------------------------------------------------------------------
def _psi_desde_proporciones(p_ref: np.ndarray, p_act: np.ndarray) -> float:
    '''PSI = suma((actual - referencia) * ln(actual / referencia)) sobre todos los grupos.'''
    p_ref = np.clip(np.asarray(p_ref, dtype=float), EPSILON, None)
    p_act = np.clip(np.asarray(p_act, dtype=float), EPSILON, None)
    return float(np.sum((p_act - p_ref) * np.log(p_act / p_ref)))


def _js_desde_proporciones(p_ref: np.ndarray, p_act: np.ndarray) -> float:
    '''
    Divergencia de Jensen-Shannon (logaritmo base 2): 0 = distribuciones iguales, 1 = totalmente
    distintas. Es simétrica y acotada; se usa como segunda opinión del PSI (no tiene umbral propio).
    scipy devuelve la *distancia* (raíz de la divergencia), por eso se eleva al cuadrado.
    '''
    return float(jensenshannon(np.asarray(p_ref, dtype=float), np.asarray(p_act, dtype=float), base=2) ** 2)


def _es_discreta(serie: pd.Series) -> bool:
    return serie.nunique(dropna=True) <= MAX_UNICOS_DISCRETA


def _tabla_proporciones_categorica(ref: pd.Series, act: pd.Series) -> pd.DataFrame:
    '''Proporciones por categoría en ambos períodos (los nulos son la categoría "Sin dato").'''
    r = ref.astype(object).where(ref.notna(), 'Sin dato').astype(str)
    a = act.astype(object).where(act.notna(), 'Sin dato').astype(str)
    tabla = pd.concat(
        [r.value_counts(normalize=True).rename('referencia'),
         a.value_counts(normalize=True).rename('actual')], axis=1,
    ).fillna(0.0)
    return tabla


def _tabla_proporciones_numerica(ref: pd.Series, act: pd.Series) -> pd.DataFrame:
    '''
    Proporciones por decil de la REFERENCIA (los cortes se fijan con la referencia, nunca con
    el período actual). Los nulos forman un grupo aparte: que cambie la tasa de faltantes
    también es drift.
    '''
    r, a = ref.dropna(), act.dropna()
    cortes = np.unique(np.quantile(r, np.linspace(0, 1, N_BINS + 1)))
    cortes[0], cortes[-1] = -np.inf, np.inf
    n_ref, n_act = len(ref), len(act)
    p_ref = list(np.histogram(r, cortes)[0] / n_ref)
    p_act = list(np.histogram(a, cortes)[0] / n_act)
    p_ref.append(ref.isna().sum() / n_ref)
    p_act.append(act.isna().sum() / n_act)
    etiquetas = [f'grupo {i + 1}' for i in range(len(cortes) - 1)] + ['Sin dato']
    return pd.DataFrame({'referencia': p_ref, 'actual': p_act}, index=etiquetas)


def tabla_proporciones(ref: pd.Series, act: pd.Series) -> pd.DataFrame:
    '''Tabla de proporciones lista para calcular el PSI o graficar la comparación.'''
    if ref.dtype.kind in 'if' and not _es_discreta(ref):
        return _tabla_proporciones_numerica(ref, act)
    return _tabla_proporciones_categorica(ref, act)


def clasificar_psi(psi: float) -> str:
    '''Semáforo del PSI.'''
    if psi >= PSI_ALTO:
        return 'Alto'
    if psi >= PSI_MODERADO:
        return 'Moderado'
    return 'Estable'


def comparar_variable(ref: pd.Series, act: pd.Series) -> dict:
    '''Compara una variable entre dos períodos: PSI, semáforo y test estadístico.'''
    tabla = tabla_proporciones(ref, act)
    psi = _psi_desde_proporciones(tabla['referencia'], tabla['actual'])

    if ref.dtype.kind in 'if' and not _es_discreta(ref):
        tipo = 'numérica'
        test = 'KS'
        p_valor = stats.ks_2samp(ref.dropna(), act.dropna()).pvalue
    else:
        tipo = 'discreta' if ref.dtype.kind in 'if' else 'categórica'
        test = 'Chi-cuadrado'
        r = ref.astype(object).where(ref.notna(), 'Sin dato').astype(str)
        a = act.astype(object).where(act.notna(), 'Sin dato').astype(str)
        cruce = pd.concat([r.value_counts(), a.value_counts()], axis=1).fillna(0)
        cruce = cruce.loc[(cruce > 0).any(axis=1)]
        p_valor = stats.chi2_contingency(cruce.values)[1] if len(cruce) > 1 else 1.0

    return {
        'tipo': tipo, 'psi': psi, 'semaforo': clasificar_psi(psi),
        'js': _js_desde_proporciones(tabla['referencia'], tabla['actual']),
        'test': test, 'p_valor': float(p_valor), 'diferencia_significativa': bool(p_valor < ALFA),
    }


# ---------------------------------------------------------------------------
# 4. DRIFT DE VARIABLES, DE PREDICCIONES Y EN EL TIEMPO
# ---------------------------------------------------------------------------
def calcular_drift(ref: pd.DataFrame, act: pd.DataFrame, columnas: list = None) -> pd.DataFrame:
    '''Una fila por variable, ordenada de mayor a menor PSI.'''
    columnas = columnas or columnas_entrada()
    filas = []
    for col in columnas:
        fila = {'variable': col}
        fila.update(comparar_variable(ref[col], act[col]))
        filas.append(fila)
    return pd.DataFrame(filas).sort_values('psi', ascending=False).reset_index(drop=True)


def drift_predicciones(paquete: dict, ref: pd.DataFrame, act: pd.DataFrame) -> dict:
    '''
    Drift del SCORE (probabilidad de no pago) y de la tasa de alertas.
    Es la señal más directa de si el modelo se está comportando distinto: no necesita saber
    si el cliente finalmente pagó (en producción esa etiqueta llega meses después).
    '''
    p_ref, p_act = proba_no_pago(paquete, ref), proba_no_pago(paquete, act)
    cortes = np.unique(np.quantile(p_ref, np.linspace(0, 1, N_BINS + 1)))
    cortes[0], cortes[-1] = -np.inf, np.inf
    prop_ref = np.histogram(p_ref, cortes)[0] / len(p_ref)
    prop_act = np.histogram(p_act, cortes)[0] / len(p_act)
    psi = _psi_desde_proporciones(prop_ref, prop_act)
    umbral = paquete['umbral_no_pago']
    return {
        'psi_score': psi, 'semaforo': clasificar_psi(psi),
        'js_score': _js_desde_proporciones(prop_ref, prop_act),
        'score_medio_ref': float(p_ref.mean()), 'score_medio_act': float(p_act.mean()),
        'tasa_alerta_ref': float((p_ref >= umbral).mean()),
        'tasa_alerta_act': float((p_act >= umbral).mean()),
        'p_ref': p_ref, 'p_act': p_act,
    }


def drift_por_periodo(paquete: dict, df: pd.DataFrame, ref: pd.DataFrame, freq: str = 'Q') -> pd.DataFrame:
    '''
    Evolución del drift: para cada trimestre (o mes, con freq='M') compara contra la referencia.
    Devuelve PSI del score, su piso de ruido (PSI esperado solo por azar con ese tamaño de
    muestra) y el PSI ajustado (PSI menos el ruido), cantidad de variables en semáforo alto,
    tasa de alertas y tasa real de no pago. Los períodos con pocas filas se marcan como no
    confiables. 'prop_en_referencia' indica qué parte del período pertenece a la referencia:
    esos períodos dan PSI bajo por construcción (los datos se comparan contra sí mismos).
    '''
    p_ref = proba_no_pago(paquete, ref)
    cortes = np.unique(np.quantile(p_ref, np.linspace(0, 1, N_BINS + 1)))
    cortes[0], cortes[-1] = -np.inf, np.inf
    prop_ref = np.histogram(p_ref, cortes)[0] / len(p_ref)
    umbral = paquete['umbral_no_pago']

    periodos = df[COLUMNA_FECHA].dt.to_period(freq)
    filas = []
    for periodo, grupo in df.groupby(periodos):
        p = proba_no_pago(paquete, grupo)
        psi = _psi_desde_proporciones(prop_ref, np.histogram(p, cortes)[0] / len(p))
        # Con pocos créditos el PSI se infla por puro azar: valor esperado ~ (grupos - 1) * (1/n + 1/N)
        piso = (len(prop_ref) - 1) * (1 / len(grupo) + 1 / len(ref))
        fila = {
            'periodo': str(periodo), 'n_creditos': len(grupo),
            'psi_score': psi, 'psi_piso_ruido': piso, 'psi_score_ajustado': max(psi - piso, 0.0),
            'prop_en_referencia': float(grupo.index.isin(ref.index).mean()),
            'tasa_alerta': float((p >= umbral).mean()),
            'tasa_no_pago_real': float((grupo[TARGET] == 0).mean()),
            'confiable': len(grupo) >= MIN_FILAS_PERIODO,
        }
        if fila['confiable']:
            tabla = calcular_drift(ref, grupo)
            fila['variables_psi_alto'] = int((tabla['psi'] >= PSI_ALTO).sum())
            fila['variables_psi_moderado'] = int(
                ((tabla['psi'] >= PSI_MODERADO) & (tabla['psi'] < PSI_ALTO)).sum())
        else:
            fila['variables_psi_alto'] = np.nan
            fila['variables_psi_moderado'] = np.nan
        filas.append(fila)
    return pd.DataFrame(filas)


def control_sin_drift() -> pd.DataFrame:
    '''
    Control del método: train vs test de la partición aleatoria estratificada oficial.
    Ambos salen de la misma población, así que el PSI debe ser ~0 en todas las variables.
    Si no lo fuera, el problema sería del método y no de los datos.
    '''
    datos = dividir_datos(cargar_base())
    return calcular_drift(datos['X_train'], datos['X_test'])


# ---------------------------------------------------------------------------
# 5. TENDENCIA, ALERTAS Y RECOMENDACIONES
# ---------------------------------------------------------------------------
ORDEN_NIVEL = {'Crítica': 0, 'Advertencia': 1, 'Sin alertas': 2}


def _num(x: float) -> str:
    '''Número con formato local: punto para los miles y coma para los decimales.'''
    if abs(x) >= 100:
        return f'{x:,.0f}'.replace(',', '.')
    return f'{x:.2f}'.replace('.', ',')


def _dec(x: float, n: int = 3) -> str:
    '''Decimal con coma.'''
    return f'{x:.{n}f}'.replace('.', ',')


def _pct(x: float) -> str:
    '''Porcentaje con coma decimal.'''
    return f'{x:.1%}'.replace('.', ',')


def describir_cambio(ref: pd.Series, act: pd.Series) -> str:
    '''Frase corta que dice QUÉ cambió en una variable (se usa en las alertas).'''
    if ref.dtype.kind in 'if' and not _es_discreta(ref):
        m_ref, m_act = ref.median(), act.median()
        texto = f'mediana {_num(m_ref)} -> {_num(m_act)}'
        if m_ref:
            texto += f' ({m_act / m_ref - 1:+.0%})'
        falt_ref, falt_act = ref.isna().mean(), act.isna().mean()
        if abs(falt_act - falt_ref) >= 0.05:
            texto += f'; faltantes {_pct(falt_ref)} -> {_pct(falt_act)}'
        return texto
    tabla = _tabla_proporciones_categorica(ref, act)
    categoria = (tabla['actual'] - tabla['referencia']).abs().idxmax()
    return (f"categoría '{categoria}': {_pct(tabla.loc[categoria, 'referencia'])} -> "
            f"{_pct(tabla.loc[categoria, 'actual'])}")


def analizar_tendencia(evolucion: pd.DataFrame) -> dict:
    '''
    Analiza la evolución del PSI del score y detecta:
      - tendencia: 'creciente' o 'decreciente' si el PSI sube (o baja) en cada uno de los últimos
        PERIODOS_TENDENCIA períodos y el cambio total supera MIN_AUMENTO_TENDENCIA;
      - cambios bruscos: períodos donde el PSI sube SALTO_ABRUPTO o más respecto del anterior.
    Se usan solo períodos confiables y posteriores a la referencia (los que están dentro de ella
    dan PSI bajo por construcción), y el PSI ajustado, que descuenta el ruido del tamaño de muestra.
    Son reglas simples y transparentes; los umbrales son constantes al inicio del archivo.
    '''
    conf = evolucion[evolucion['confiable'] & (evolucion['prop_en_referencia'] < 0.5)].reset_index(drop=True)
    if len(conf) < 3:
        return {'tendencia': 'sin datos suficientes', 'cambios_abruptos': [], 'gravedad_abrupto': 'Sin alertas',
                'mensaje': 'Hay menos de 3 períodos posteriores a la referencia con datos suficientes '
                           'para analizar la tendencia.'}

    psi = conf['psi_score_ajustado'].to_numpy()
    indices_abruptos = [i + 1 for i, salto in enumerate(np.diff(psi)) if salto >= SALTO_ABRUPTO]
    abruptos = [conf.loc[i, 'periodo'] for i in indices_abruptos]
    # Un salto es grave solo si llega a un nivel de drift alto; si no, es una advertencia
    gravedad = 'Sin alertas'
    if indices_abruptos:
        gravedad = 'Crítica' if max(psi[i] for i in indices_abruptos) >= PSI_ALTO else 'Advertencia'

    ultimos = psi[-PERIODOS_TENDENCIA:]
    pasos = np.diff(ultimos)
    if (pasos > 0).all() and ultimos[-1] - ultimos[0] >= MIN_AUMENTO_TENDENCIA:
        tendencia = 'creciente'
    elif (pasos < 0).all() and ultimos[0] - ultimos[-1] >= MIN_AUMENTO_TENDENCIA:
        tendencia = 'decreciente'
    else:
        tendencia = 'sin tendencia clara'

    mensaje = (f'Tendencia del PSI del score (sin el ruido de muestra, períodos posteriores a la referencia): '
               f'{tendencia} ({_dec(ultimos[0])} -> {_dec(ultimos[-1])} en los últimos {len(ultimos)} períodos). ')
    if abruptos:
        mensaje += f'Cambio brusco (aumento >= {_dec(SALTO_ABRUPTO, 2)}) en: {", ".join(abruptos)}.'
    else:
        mensaje += f'Sin cambios bruscos (aumento >= {_dec(SALTO_ABRUPTO, 2)} entre períodos seguidos).'
    return {'tendencia': tendencia, 'cambios_abruptos': abruptos, 'gravedad_abrupto': gravedad,
            'mensaje': mensaje}


def generar_alertas(tabla: pd.DataFrame, pred: dict, tendencia: dict,
                    ref: pd.DataFrame, act: pd.DataFrame) -> list:
    '''
    Convierte las métricas en mensajes automáticos con una acción sugerida.
    Devuelve una lista de dicts (nivel, titulo, detalle, accion) ordenada de más a menos grave.
    Niveles: 'Crítica' (revisar o reentrenar), 'Advertencia' (vigilar) y 'Sin alertas'.
    '''
    alertas = []

    def agregar(nivel, titulo, detalle, accion):
        alertas.append({'nivel': nivel, 'titulo': titulo, 'detalle': detalle, 'accion': accion})

    def detalle_variables(filas):
        return '\n'.join(
            f'- {f.variable}: PSI {_dec(f.psi)} ({describir_cambio(ref[f.variable], act[f.variable])})'
            for f in filas.itertuples()
        )

    altas = tabla[tabla['semaforo'] == 'Alto']
    moderadas = tabla[tabla['semaforo'] == 'Moderado']
    if len(altas):
        agregar('Crítica', f'{len(altas)} variable(s) con cambio alto (PSI >= {_dec(PSI_ALTO, 2)})',
                detalle_variables(altas),
                'Revisar el origen del cambio (fuente de datos, política de crédito, calidad del dato) '
                'y evaluar reentrenar el modelo con datos recientes o revisar esas variables.')
    if len(moderadas):
        agregar('Advertencia',
                f'{len(moderadas)} variable(s) con cambio moderado '
                f'(PSI entre {_dec(PSI_MODERADO, 2)} y {_dec(PSI_ALTO, 2)})',
                detalle_variables(moderadas),
                'Vigilar: confirmar si el cambio es esperado (producto, canal, estacionalidad). Si persiste '
                'o crece, revisar el aporte de esas variables al modelo.')

    psi_score = pred['psi_score']
    if psi_score >= PSI_ALTO:
        agregar('Crítica', f'El score del modelo cambió mucho (PSI {_dec(psi_score)})', '',
                'Reentrenar el modelo con datos recientes antes de seguir usando el umbral actual.')
    elif psi_score >= PSI_MODERADO:
        agregar('Advertencia', f'El score del modelo cambió de forma moderada (PSI {_dec(psi_score)})', '',
                'Acortar la periodicidad del monitoreo y preparar el reentrenamiento.')

    t_ref, t_act = pred['tasa_alerta_ref'], pred['tasa_alerta_act']
    if t_ref > 0 and abs(t_act / t_ref - 1) >= CAMBIO_RELATIVO_ALERTAS:
        agregar('Advertencia',
                f'La tasa de alertas del modelo pasó de {_pct(t_ref)} a {_pct(t_act)} '
                f'({t_act / t_ref - 1:+.0%} en términos relativos)', '',
                'Revisar si el umbral de decisión sigue siendo adecuado para la capacidad de revisión '
                'del negocio y qué variables están empujando el score.')

    if tendencia['cambios_abruptos']:
        agregar(tendencia['gravedad_abrupto'],
                'Cambio brusco del drift en ' + ', '.join(tendencia['cambios_abruptos']),
                tendencia['mensaje'],
                'Investigar qué ocurrió en ese período (cambio de producto, de canal o de fuente de datos) '
                'antes de seguir usando el modelo sin cambios.')
    elif tendencia['tendencia'] == 'creciente':
        agregar('Advertencia', 'El drift del score crece período a período', tendencia['mensaje'],
                f'Planificar el reentrenamiento con datos recientes: si el PSI del score llega a '
                f'{_dec(PSI_ALTO, 2)}, pasa a ser una alerta crítica.')

    if not alertas:
        agregar('Sin alertas', 'No se detectaron desviaciones relevantes', '',
                'Mantener el monitoreo con la periodicidad definida.')
    return sorted(alertas, key=lambda a: ORDEN_NIVEL[a['nivel']])


def nivel_general(alertas: list) -> str:
    '''Nivel más grave entre las alertas: 'Crítica', 'Advertencia' o 'Sin alertas'.'''
    return alertas[0]['nivel']


def recomendacion_general(nivel: str) -> str:
    '''Recomendación de una línea según el nivel general.'''
    return {
        'Crítica': 'Revisar el modelo y reentrenarlo con datos recientes.',
        'Advertencia': 'Mantener el modelo en uso y vigilarlo de cerca; preparar el reentrenamiento '
                       'si el drift sigue creciendo.',
        'Sin alertas': 'Sin acción: el modelo opera sobre una población similar a la de entrenamiento.',
    }[nivel]


def resumen_estado(tabla: pd.DataFrame, pred: dict, nivel: str) -> str:
    '''Frase corta con el estado general, para el tablero y el README.'''
    n_alto = int((tabla['semaforo'] == 'Alto').sum())
    n_mod = int((tabla['semaforo'] == 'Moderado').sum())
    estado = {'Crítica': 'ALERTA: revisar el modelo',
              'Advertencia': 'VIGILAR: hay cambios que seguir de cerca',
              'Sin alertas': 'ESTABLE: sin cambios relevantes'}[nivel]
    return (f'{estado} | variables en PSI alto: {n_alto}, moderado: {n_mod} de {len(tabla)} | '
            f'PSI del score: {pred["psi_score"]:.3f} ({pred["semaforo"]})')


# ---------------------------------------------------------------------------
# 6. TABLA DE MONITOREO (datos + pronósticos) Y EJECUCIÓN
# ---------------------------------------------------------------------------
def tabla_datos_pronosticos(paquete: dict, df: pd.DataFrame) -> pd.DataFrame:
    '''
    Tabla de monitoreo: las variables de entrada (ya limpias) junto con el pronóstico del modelo
    (probabilidad de no pago y alerta) y el resultado real (1 = pagó, 0 = no pagó).
    En producción, esta tabla se armaría con las predicciones que el modelo fue entregando.
    '''
    probabilidad = proba_no_pago(paquete, df)
    salida = df[[COLUMNA_FECHA] + paquete['columnas_entrada']].copy()
    salida['prob_no_pago'] = probabilidad
    salida['alerta_modelo'] = (probabilidad >= paquete['umbral_no_pago']).astype(int)
    salida['pago_real'] = df[TARGET].to_numpy()
    return salida


def main():
    '''Calcula el drift (referencia vs actual), muestra alertas y recomendaciones y guarda CSV.'''
    paquete = cargar_modelo()
    df = cargar_base()
    ref, act = separar_periodos(df)

    print(f'Referencia: {len(ref):,} créditos ({ref[COLUMNA_FECHA].min():%d/%m/%Y} a {ref[COLUMNA_FECHA].max():%d/%m/%Y})')
    print(f'Actual:     {len(act):,} créditos ({act[COLUMNA_FECHA].min():%d/%m/%Y} a {act[COLUMNA_FECHA].max():%d/%m/%Y})\n')

    tabla = calcular_drift(ref, act)
    pred = drift_predicciones(paquete, ref, act)
    evolucion = drift_por_periodo(paquete, df, ref)
    tendencia = analizar_tendencia(evolucion)
    alertas = generar_alertas(tabla, pred, tendencia, ref, act)
    nivel = nivel_general(alertas)

    print(tabla.round(4).to_string(index=False))
    print(f'\nScore: media {pred["score_medio_ref"]:.3f} -> {pred["score_medio_act"]:.3f} | '
          f'alertas {pred["tasa_alerta_ref"]:.1%} -> {pred["tasa_alerta_act"]:.1%} | '
          f'PSI {pred["psi_score"]:.3f} | JS {pred["js_score"]:.4f}')
    print(resumen_estado(tabla, pred, nivel))

    control = control_sin_drift()
    print(f'\nControl (train vs test aleatorio): PSI máximo = {control["psi"].max():.4f} '
          f'(esperado ~0; ningún semáforo distinto de Estable: {(control["semaforo"] != "Estable").sum() == 0})')

    print('\nEvolución trimestral:')
    print(evolucion.round(4).to_string(index=False))
    print(f'\n{tendencia["mensaje"]}')

    print('\nALERTAS Y RECOMENDACIONES')
    for alerta in alertas:
        print(f'[{alerta["nivel"].upper()}] {alerta["titulo"]}')
        if alerta['detalle']:
            print(alerta['detalle'])
        print(f'  -> {alerta["accion"]}\n')
    print(f'Recomendación general: {recomendacion_general(nivel)}')

    RUTA_REPORTES.mkdir(exist_ok=True)
    tabla.to_csv(RUTA_REPORTES / 'drift_variables.csv', index=False)
    evolucion.to_csv(RUTA_REPORTES / 'drift_evolucion.csv', index=False)
    pd.DataFrame(alertas).to_csv(RUTA_REPORTES / 'drift_alertas.csv', index=False)
    print(f'\nReportes guardados en: {RUTA_REPORTES}')


if __name__ == '__main__':
    main()
