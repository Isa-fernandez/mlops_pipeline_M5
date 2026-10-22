'''
app_streamlit.py
----------------
PIM5 · Avance 3 · Tablero en Streamlit: monitoreo de data drift, alertas y simulador de riesgo.

Qué muestra (una pestaña por tema):
    1. Estado general: semáforo, indicadores y tabla de métricas de drift por variable
       (PSI, Jensen-Shannon, KS y chi-cuadrado).
    2. Detalle por variable: distribución histórica vs actual y drift del score del modelo.
    3. Evolución: drift a lo largo del tiempo, con detección de tendencias y cambios bruscos.
    4. Alertas y recomendaciones: mensajes automáticos y sugerencias (reentrenar, revisar variables).
    5. Datos y pronósticos: tabla de monitoreo (datos + probabilidad de no pago + resultado real).
    6. Simulador: probabilidad de no pago de un cliente nuevo y predicción por lotes (CSV),
       consultando la API de FastAPI (model_deploy.py).

Toda la lógica de cálculo vive en model_monitoring.py; este archivo solo la muestra.
Avance 4: el simulador consulta la API (por defecto http://localhost:8000; se puede cambiar con
la variable de entorno API_URL). Si la API no está en línea, usa el modelo local y lo avisa.

Ejecutar (desde la raíz del repo, con el entorno virtual activo):
    1. API (otra terminal):  uvicorn src.model_deploy:app   (o el contenedor de Docker)
    2. App:                  streamlit run src/app_streamlit.py
'''

import json
import os

import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title='Monitoreo del modelo de riesgo', layout='wide')

from ft_engineering import COLUMNA_FECHA, columnas_entrada  # noqa: E402
from model_monitoring import (  # noqa: E402
    CAMBIO_RELATIVO_ALERTAS, MIN_AUMENTO_TENDENCIA, PERIODOS_TENDENCIA, PSI_ALTO, PSI_MODERADO,
    SALTO_ABRUPTO, analizar_tendencia, calcular_drift, cargar_base, cargar_modelo, control_sin_drift,
    drift_por_periodo, drift_predicciones, generar_alertas, nivel_general, proba_no_pago,
    recomendacion_general, resumen_estado, separar_periodos, tabla_datos_pronosticos,
    tabla_proporciones,
)

COLORES = {'Estable': '#2E8B57', 'Moderado': '#E6A100', 'Alto': '#C0392B'}
COLOR_REF, COLOR_ACT = '#7F8C8D', '#2E86AB'
AVISO_POR_NIVEL = {'Crítica': st.error, 'Advertencia': st.warning, 'Sin alertas': st.success}
API_URL = os.getenv('API_URL', 'http://localhost:8000')   # dirección de la API (model_deploy.py)
TIEMPO_ESPERA_API = 30                                    # segundos máximos de espera


def api_en_linea() -> bool:
    '''True si la API responde en /health.'''
    try:
        return requests.get(f'{API_URL}/health', timeout=2).ok
    except requests.RequestException:
        return False


def predecir_con_api(registros: pd.DataFrame) -> dict:
    '''
    Envía un lote de registros a POST /predict y devuelve la respuesta de la API.
    Se pasa por to_json para convertir los NaN en null y los tipos de numpy en tipos de JSON.
    '''
    datos = json.loads(registros.to_json(orient='records'))
    respuesta = requests.post(f'{API_URL}/predict', json={'data': datos}, timeout=TIEMPO_ESPERA_API)
    respuesta.raise_for_status()
    return respuesta.json()


# ---------------------------------------------------------------------------
# DATOS (con caché: se calculan una vez y no en cada clic)
# ---------------------------------------------------------------------------
@st.cache_resource
def obtener_modelo():
    return cargar_modelo()


@st.cache_data
def obtener_base():
    return cargar_base()


@st.cache_data
def calcular_todo(prop_referencia: float, freq: str):
    '''Calcula todas las tablas del tablero para una referencia y una periodicidad dadas.'''
    paquete, df = obtener_modelo(), obtener_base()
    ref, act = separar_periodos(df, prop_referencia)
    tabla = calcular_drift(ref, act)
    pred = drift_predicciones(paquete, ref, act)
    evolucion = drift_por_periodo(paquete, df, ref, freq)
    tendencia = analizar_tendencia(evolucion)
    alertas = generar_alertas(tabla, pred, tendencia, ref, act)
    return ref, act, tabla, pred, evolucion, tendencia, alertas


@st.cache_data
def obtener_control():
    return control_sin_drift()


@st.cache_data
def obtener_datos_pronosticos():
    return tabla_datos_pronosticos(obtener_modelo(), obtener_base())


try:
    paquete = obtener_modelo()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

df = obtener_base()

# ---------------------------------------------------------------------------
# BARRA LATERAL
# ---------------------------------------------------------------------------
st.sidebar.header('Configuración')
prop = st.sidebar.slider(
    'Proporción de créditos usada como referencia (los más antiguos)', 0.50, 0.90, 0.80, 0.05,
    help='El resto (los más recientes) se compara contra esa referencia.',
)
periodicidad = st.sidebar.radio(
    'Periodicidad del análisis temporal', ['Trimestral', 'Mensual'], horizontal=True,
    help='Cada período se compara contra la referencia. Los períodos con menos de 100 créditos se excluyen.',
)
freq = {'Trimestral': 'Q', 'Mensual': 'M'}[periodicidad]
etiqueta_periodo = 'Trimestre' if freq == 'Q' else 'Mes'

ref, act, tabla, pred, evolucion, tendencia, alertas = calcular_todo(prop, freq)
nivel = nivel_general(alertas)

st.sidebar.markdown(
    f'**Referencia:** {len(ref):,} créditos  \n'
    f'{ref[COLUMNA_FECHA].min():%d/%m/%Y} a {ref[COLUMNA_FECHA].max():%d/%m/%Y}\n\n'
    f'**Actual:** {len(act):,} créditos  \n'
    f'{act[COLUMNA_FECHA].min():%d/%m/%Y} a {act[COLUMNA_FECHA].max():%d/%m/%Y}'
)
st.sidebar.caption(
    f'Modelo: {paquete["nombre_modelo"]} · umbral de no pago: {paquete["umbral_no_pago"]:.3f}'
)

st.title('Monitoreo del modelo de riesgo crediticio')
st.caption('Compara los créditos recientes con los usados como referencia para detectar data drift.')

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    'Estado general', 'Detalle por variable', 'Evolución',
    'Alertas y recomendaciones', 'Datos y pronósticos', 'Simulador',
])

# ---------------------------------------------------------------------------
# TAB 1 · ESTADO GENERAL
# ---------------------------------------------------------------------------
with tab1:
    n_alto = int((tabla['semaforo'] == 'Alto').sum())
    n_mod = int((tabla['semaforo'] == 'Moderado').sum())

    AVISO_POR_NIVEL[nivel](resumen_estado(tabla, pred, nivel))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Variables con PSI alto', f'{n_alto} de {len(tabla)}')
    c2.metric('Variables con PSI moderado', f'{n_mod} de {len(tabla)}')
    c3.metric('PSI del score', f'{pred["psi_score"]:.3f}', pred['semaforo'], delta_color='off')
    c4.metric(
        'Tasa de alertas', f'{pred["tasa_alerta_act"]:.1%}',
        f'{pred["tasa_alerta_act"] - pred["tasa_alerta_ref"]:+.1%} vs referencia', delta_color='off',
    )
    st.progress(
        min(pred['psi_score'] / PSI_ALTO, 1.0),
        text=f'Barra de riesgo del score: PSI {pred["psi_score"]:.3f} sobre el nivel crítico de {PSI_ALTO}',
    )

    grafico = tabla.sort_values('psi')
    fig = px.bar(
        grafico, x='psi', y='variable', orientation='h', color='semaforo',
        color_discrete_map=COLORES, category_orders={'semaforo': ['Estable', 'Moderado', 'Alto']},
        title='PSI por variable (referencia vs actual)', labels={'psi': 'PSI', 'variable': ''},
    )
    fig.add_vline(x=PSI_MODERADO, line_dash='dash', line_color='#E6A100')
    fig.add_vline(x=PSI_ALTO, line_dash='dash', line_color='#C0392B')
    fig.update_layout(height=620, legend_title_text='Semáforo')
    st.plotly_chart(fig, width='stretch')

    with st.expander('Cómo leer las métricas'):
        st.markdown(
            f'- **PSI** (Population Stability Index): **< {PSI_MODERADO}** estable, **{PSI_MODERADO} a {PSI_ALTO}** '
            f'cambio moderado (vigilar), **≥ {PSI_ALTO}** cambio alto (revisar el modelo). Decide el semáforo.\n'
            '- **JS** (divergencia de Jensen-Shannon): va de 0 (distribuciones iguales) a 1 (totalmente distintas). '
            'Es una segunda opinión del PSI y no tiene umbral propio.\n'
            '- **KS** (numéricas) y **chi-cuadrado** (categóricas): tests estadísticos. El **p-valor** indica si '
            'la diferencia es detectable, pero con miles de filas casi todo es "significativo", por eso no '
            'decide el semáforo.\n'
            '- Los nulos cuentan como un grupo más: si cambia la proporción de datos faltantes, también es drift.'
        )
    control = obtener_control()
    st.caption(
        f'Control del método: comparando train vs test (misma población) el PSI máximo es '
        f'{control["psi"].max():.4f}; el método no inventa drift.'
    )
    st.dataframe(
        tabla.assign(
            psi=tabla['psi'].round(4), js=tabla['js'].round(4), p_valor=tabla['p_valor'].round(4),
        ).rename(columns={'js': 'JS', 'diferencia_significativa': 'significativo (p<0,01)'}),
        width='stretch', hide_index=True,
    )

# ---------------------------------------------------------------------------
# TAB 2 · DETALLE POR VARIABLE
# ---------------------------------------------------------------------------
with tab2:
    variable = st.selectbox('Variable', tabla['variable'].tolist())
    fila = tabla[tabla['variable'] == variable].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('PSI', f'{fila["psi"]:.3f}', fila['semaforo'], delta_color='off')
    c2.metric('JS', f'{fila["js"]:.4f}')
    c3.metric('Test', fila['test'])
    c4.metric('p-valor', f'{fila["p_valor"]:.4f}')

    prop_tabla = tabla_proporciones(ref[variable], act[variable]).reset_index(names='grupo')
    largo = prop_tabla.melt(id_vars='grupo', var_name='período', value_name='proporción')
    fig = px.bar(
        largo, x='grupo', y='proporción', color='período', barmode='group',
        color_discrete_map={'referencia': COLOR_REF, 'actual': COLOR_ACT},
        title=f'{variable}: distribución histórica (referencia) vs actual',
        labels={'grupo': 'Grupo (deciles de la referencia o categorías)'},
    )
    fig.update_yaxes(tickformat='.0%')
    st.plotly_chart(fig, width='stretch')
    st.caption('En variables numéricas los grupos son los deciles de la referencia: cada uno '
               'contiene ~10 % de la referencia. Si el actual se aleja de 10 %, hay drift.')

    st.subheader('Drift del score del modelo')
    score = pd.concat([
        pd.DataFrame({'probabilidad de no pago': pred['p_ref'], 'período': 'referencia'}),
        pd.DataFrame({'probabilidad de no pago': pred['p_act'], 'período': 'actual'}),
    ])
    fig = px.histogram(
        score, x='probabilidad de no pago', color='período', barmode='overlay', histnorm='probability',
        nbins=40, opacity=0.6, color_discrete_map={'referencia': COLOR_REF, 'actual': COLOR_ACT},
    )
    fig.add_vline(x=paquete['umbral_no_pago'], line_dash='dash', annotation_text='umbral')
    st.plotly_chart(fig, width='stretch')
    st.caption(
        f'Score medio: {pred["score_medio_ref"]:.3f} (referencia) → {pred["score_medio_act"]:.3f} (actual). '
        f'Tasa de alertas: {pred["tasa_alerta_ref"]:.1%} → {pred["tasa_alerta_act"]:.1%}. '
        f'PSI del score: {pred["psi_score"]:.3f} · JS: {pred["js_score"]:.4f}.'
    )

# ---------------------------------------------------------------------------
# TAB 3 · EVOLUCIÓN
# ---------------------------------------------------------------------------
with tab3:
    conf = evolucion[evolucion['confiable']]
    excluidos = evolucion[~evolucion['confiable']]
    if len(excluidos):
        st.warning(
            'Se excluyen por tener muy pocos créditos (PSI no confiable): '
            + ', '.join(f'{p} ({n} créditos)' for p, n in zip(excluidos['periodo'], excluidos['n_creditos']))
        )

    if tendencia['cambios_abruptos']:
        AVISO_POR_NIVEL[tendencia['gravedad_abrupto']](tendencia['mensaje'])
    elif tendencia['tendencia'] == 'creciente':
        st.warning(tendencia['mensaje'])
    else:
        st.info(tendencia['mensaje'])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=conf['periodo'], y=conf['psi_score'], mode='lines+markers',
                             name='PSI del score', line=dict(color=COLOR_ACT)))
    fig.add_trace(go.Scatter(x=conf['periodo'], y=conf['psi_piso_ruido'], mode='lines',
                             name='Ruido esperado por azar', line=dict(color=COLOR_REF, dash='dot')))
    fig.add_hline(y=PSI_MODERADO, line_dash='dash', line_color='#E6A100')
    fig.add_hline(y=PSI_ALTO, line_dash='dash', line_color='#C0392B')
    fig.update_layout(title=f'PSI del score por {etiqueta_periodo.lower()} (contra la referencia)',
                      yaxis_title='PSI', xaxis_title=etiqueta_periodo)
    st.plotly_chart(fig, width='stretch')

    c1, c2 = st.columns(2)
    with c1:
        barras = conf.melt(id_vars='periodo', value_vars=['variables_psi_moderado', 'variables_psi_alto'],
                           var_name='nivel', value_name='variables')
        barras['nivel'] = barras['nivel'].map(
            {'variables_psi_moderado': 'Moderado', 'variables_psi_alto': 'Alto'})
        fig = px.bar(barras, x='periodo', y='variables', color='nivel', barmode='stack',
                     color_discrete_map=COLORES, title=f'Variables con drift por {etiqueta_periodo.lower()}',
                     labels={'periodo': etiqueta_periodo, 'variables': 'Cantidad de variables'})
        st.plotly_chart(fig, width='stretch')
    with c2:
        tasas = conf.melt(id_vars='periodo', value_vars=['tasa_alerta', 'tasa_no_pago_real'],
                          var_name='serie', value_name='tasa')
        tasas['serie'] = tasas['serie'].map(
            {'tasa_alerta': 'Alertas del modelo', 'tasa_no_pago_real': 'No pago real'})
        fig = px.line(tasas, x='periodo', y='tasa', color='serie', markers=True,
                      title='Alertas del modelo y tasa real de no pago',
                      labels={'periodo': etiqueta_periodo, 'tasa': 'Tasa'})
        fig.update_yaxes(tickformat='.0%')
        st.plotly_chart(fig, width='stretch')

    st.caption(
        'Los períodos que están dentro de la referencia (por ejemplo 2025Q2) muestran poco drift por '
        'construcción, por eso la tendencia y los cambios bruscos se calculan solo con períodos '
        'posteriores. Con pocos créditos el PSI sube por puro azar (línea punteada: valor esperado sin '
        'drift real); la detección descuenta ese ruido y los conteos de variables de abajo no, así que '
        'conviene leerlos con cautela en los períodos chicos. La tasa real de no pago solo se conoce '
        'cuando vence el crédito; en producción llega con retraso y por eso el monitoreo se apoya en el '
        'drift de datos y del score.'
    )
    st.dataframe(evolucion.round(4), width='stretch', hide_index=True)

# ---------------------------------------------------------------------------
# TAB 4 · ALERTAS Y RECOMENDACIONES
# ---------------------------------------------------------------------------
with tab4:
    st.subheader('Alertas automáticas')
    for alerta in alertas:
        cuerpo = f'**{alerta["titulo"]}**'
        if alerta['detalle']:
            cuerpo += f'\n\n{alerta["detalle"]}'
        cuerpo += f'\n\n**Qué hacer:** {alerta["accion"]}'
        AVISO_POR_NIVEL[alerta['nivel']](cuerpo)

    st.subheader('Recomendación general')
    st.markdown(recomendacion_general(nivel))

    with st.expander('Reglas que disparan las alertas'):
        st.markdown(
            f'- **Variable** con PSI ≥ {PSI_ALTO}: alerta crítica. Entre {PSI_MODERADO} y {PSI_ALTO}: advertencia.\n'
            f'- **Score del modelo**: mismos umbrales de PSI.\n'
            f'- **Tasa de alertas** del modelo con variación relativa ≥ {CAMBIO_RELATIVO_ALERTAS:.0%}: advertencia.\n'
            f'- **Tendencia**: el PSI del score sube en cada uno de los últimos {PERIODOS_TENDENCIA} períodos '
            f'(aumento total ≥ {MIN_AUMENTO_TENDENCIA}): advertencia.\n'
            f'- **Cambio brusco**: el PSI del score sube {SALTO_ABRUPTO} o más entre dos períodos seguidos: '
            f'alerta crítica.\n\n'
            'Los umbrales de PSI son el estándar de la industria; los demás son reglas simples que conviene '
            'ajustar con el negocio. Los umbrales están al inicio de `model_monitoring.py`.'
        )

# ---------------------------------------------------------------------------
# TAB 5 · DATOS Y PRONÓSTICOS
# ---------------------------------------------------------------------------
with tab5:
    datos = obtener_datos_pronosticos()
    st.caption('Tabla de monitoreo: variables de entrada (ya limpias), pronóstico del modelo y resultado real '
               '(pago_real: 1 = pagó, 0 = no pagó).')
    ventana = st.radio('Ventana', ['Período actual', 'Período de referencia', 'Toda la base'], horizontal=True)
    indices = {'Período actual': act.index, 'Período de referencia': ref.index, 'Toda la base': datos.index}[ventana]
    vista = datos.loc[indices].sort_values(COLUMNA_FECHA, ascending=False)

    c1, c2, c3 = st.columns(3)
    c1.metric('Créditos', f'{len(vista):,}')
    c2.metric('Tasa de alertas del modelo', f'{vista["alerta_modelo"].mean():.1%}')
    c3.metric('Tasa real de no pago', f'{(vista["pago_real"] == 0).mean():.1%}')

    st.dataframe(vista, width='stretch', hide_index=True)
    st.download_button('Descargar CSV', data=vista.to_csv(index=False).encode('utf-8-sig'),
                       file_name='datos_y_pronosticos.csv', mime='text/csv')

# ---------------------------------------------------------------------------
# TAB 6 · SIMULADOR
# ---------------------------------------------------------------------------
with tab6:
    st.subheader('Probabilidad de no pago de un cliente nuevo')
    en_linea = api_en_linea()
    if en_linea:
        st.success(f'Conectado a la API: {API_URL} (documentación en {API_URL}/docs)')
    else:
        st.warning(f'La API no responde en {API_URL}: se usa el modelo local. Para usar la API, '
                   'iniciarla con `uvicorn src.model_deploy:app` o con el contenedor de Docker.')
    st.caption('Las variables que no se piden aquí se completan con el valor típico (mediana o moda) '
               'de la referencia.')

    tipicos = {}
    for col in columnas_entrada():
        serie = ref[col]
        tipicos[col] = serie.median() if serie.dtype.kind in 'if' else serie.mode(dropna=True).iloc[0]

    c1, c2, c3 = st.columns(3)
    capital = c1.number_input('Capital prestado', 0.0, 1e10, float(tipicos['capital_prestado']), 100000.0)
    plazo = c1.number_input('Plazo (meses)', 1, 120, int(tipicos['plazo_meses']))
    cuota = c1.number_input('Cuota pactada', 0.0, 1e9, float(tipicos['cuota_pactada']), 10000.0)
    salario = c2.number_input('Salario del cliente', 0.0, 1e10, float(tipicos['salario_cliente']), 100000.0)
    otros = c2.number_input('Total otros préstamos', 0.0, 1e10, float(tipicos['total_otros_prestamos']), 100000.0)
    edad = c2.number_input('Edad', 18, 90, int(tipicos['edad_cliente']))
    puntaje_dc = c3.number_input('Puntaje Datacrédito', 1.0, 1000.0, float(tipicos['puntaje_datacredito']))
    vigentes = c3.number_input('Créditos vigentes', 0, 50, int(tipicos['cant_creditosvigentes']))
    tipo_cred = c3.selectbox('Tipo de crédito', ['4', '9', '10', '6', 'Otros'])
    tipo_lab = c1.selectbox('Tipo laboral', sorted(df['tipo_laboral'].dropna().unique()))
    tendencia_ing = c2.selectbox('Tendencia de ingresos', ['Creciente', 'Estable', 'Decreciente', 'Sin dato'])

    if st.button('Calcular riesgo', type='primary'):
        cliente = {**tipicos,
                   'capital_prestado': capital, 'plazo_meses': plazo, 'cuota_pactada': cuota,
                   'salario_cliente': salario, 'total_otros_prestamos': otros, 'edad_cliente': edad,
                   'puntaje_datacredito': puntaje_dc, 'cant_creditosvigentes': vigentes,
                   'tipo_credito': tipo_cred, 'tipo_laboral': tipo_lab,
                   'tendencia_ingresos': None if tendencia_ing == 'Sin dato' else tendencia_ing}
        entrada = pd.DataFrame([cliente])[columnas_entrada()]
        if en_linea:
            resultado = predecir_con_api(entrada)['resultados'][0]
            probabilidad, alerta_cliente, fuente = resultado['prob_no_pago'], resultado['alerta'], 'API'
        else:
            probabilidad = float(proba_no_pago(paquete, entrada)[0])
            alerta_cliente = probabilidad >= paquete['umbral_no_pago']
            fuente = 'modelo local'
        m1, m2, m3 = st.columns(3)
        m1.metric('Probabilidad de no pago', f'{probabilidad:.1%}')
        m2.metric('Decisión del modelo', 'ALERTA: revisar' if alerta_cliente else 'Sin alerta')
        m3.metric('Calculado con', fuente)
        st.caption(f'La alerta se activa con probabilidad ≥ {paquete["umbral_no_pago"]:.1%}. '
                   'El modelo tiene capacidad de discriminación limitada (ROC-AUC ≈ 0,68): '
                   'sirve para priorizar revisiones, no para decidir solo.')

    st.divider()
    st.subheader('Predicción por lotes (CSV) con la API')
    st.caption('Subir un CSV con las 20 columnas del modelo (las columnas de más se ignoran). '
               'Cada fila se envía a la API en una sola solicitud por lote.')
    archivo = st.file_uploader('Archivo CSV', type='csv', disabled=not en_linea)
    if archivo is not None and en_linea:
        lote = pd.read_csv(archivo)
        faltantes = [c for c in columnas_entrada() if c not in lote.columns]
        if faltantes:
            st.error(f'Faltan columnas: {faltantes}')
        else:
            respuesta = predecir_con_api(lote[columnas_entrada()])
            salida = pd.concat([lote.reset_index(drop=True), pd.DataFrame(respuesta['resultados'])], axis=1)
            c1, c2 = st.columns(2)
            c1.metric('Registros', f'{respuesta["n_registros"]:,}')
            c2.metric('Alertas', f'{respuesta["n_alertas"]:,} ({respuesta["n_alertas"] / respuesta["n_registros"]:.1%})')
            st.dataframe(salida, width='stretch', hide_index=True)
            st.download_button('Descargar resultados', data=salida.to_csv(index=False).encode('utf-8-sig'),
                               file_name='predicciones_api.csv', mime='text/csv')
