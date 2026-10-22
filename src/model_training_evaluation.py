'''
model_training_evaluation.py
----------------------------
PIM5 · Avance 2 (V1.1.1) · Entrenamiento, evaluación y selección del modelo de riesgo crediticio.

Qué hace este módulo:
    1. Define los modelos candidatos (con el desbalanceo 95/5 ya tratado).
    2. Los evalúa con métricas de la clase de interés (la 'no pagó', Pago_atiempo = 0).
    3. Compara los modelos con partición train/test y con validación cruzada estratificada.
    4. Elige el mejor, define su umbral de decisión y lo guarda para la API (Avance 4).

Convención de métricas (importante):
    La clase de interés es la 0 ('no pagó a tiempo'), que es la minoritaria (4,75 %).
    Precision, Recall y F1 se calculan sobre esa clase. La 'exactitud' (accuracy) se informa
    solo como referencia: un modelo que prediga 'todos pagan' ya tiene 95,25 %.
    ROC-AUC y PR-AUC se calculan con la probabilidad de NO pago.

Uso desde la terminal (ejecuta todo y guarda modelo_final.joblib):
    python model_training_evaluation.py
'''

import time
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, make_scorer,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from ft_engineering import (
    SEMILLA, columnas_entrada, crear_pipeline_features, ft_engineering,
)

warnings.filterwarnings('ignore')

RUTA_SRC = Path(__file__).resolve().parent
RUTA_MODELO = RUTA_SRC.parent / 'models' / 'modelo_final.joblib'
N_FOLDS = 5
UMBRAL_DEFECTO = 0.5


# ---------------------------------------------------------------------------
# 1. MODELOS CANDIDATOS
# ---------------------------------------------------------------------------
def crear_modelos(y_train: pd.Series) -> dict:
    '''
    Devuelve {nombre: (clasificador, escalar)}.
    'escalar' indica qué pipeline de características usa cada modelo (ver ft_engineering).

    Los hiperparámetros son deliberadamente conservadores (árboles poco profundos, hojas con
    muchas filas, regularización): con solo ~400 casos de 'no pagó' en entrenamiento, modelos
    más flexibles memorizan el train y no mejoran en validación (ver notebook, sección 11).

    Tratamiento del desbalanceo (95/5): se pondera la clase minoritaria.
        - class_weight='balanced' en scikit-learn y LightGBM.
        - scale_pos_weight en XGBoost: su 'clase positiva' es la 1 (pagó), por lo que el
          cociente n_no_pagan / n_pagan (< 1) le resta peso a la 1, lo que equivale a
          reforzar la clase 0. Se calcula con y_train, no se escribe a mano.
    '''
    n_no_pago = int((y_train == 0).sum())
    n_pago = int((y_train == 1).sum())

    return {
        'Referencia (prior)': (DummyClassifier(strategy='prior'), False),
        'Regresión Logística': (LogisticRegression(
            class_weight='balanced', max_iter=1000, random_state=SEMILLA), True),
        'Árbol de Decisión': (DecisionTreeClassifier(
            max_depth=4, min_samples_leaf=50, class_weight='balanced', random_state=SEMILLA), False),
        'Random Forest': (RandomForestClassifier(
            n_estimators=300, max_depth=5, min_samples_leaf=50, max_features='sqrt',
            class_weight='balanced_subsample', n_jobs=-1, random_state=SEMILLA), False),
        'XGBoost': (XGBClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.03, min_child_weight=2, reg_lambda=5,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=n_no_pago / n_pago, eval_metric='logloss', n_jobs=-1,
            random_state=SEMILLA), False),
        'LightGBM': (LGBMClassifier(
            n_estimators=150, num_leaves=6, learning_rate=0.03, min_child_samples=50, reg_lambda=5,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, class_weight='balanced',
            verbose=-1, n_jobs=-1, random_state=SEMILLA), False),
    }


def construir_pipeline(clasificador, escalar: bool, con_puntaje: bool = False) -> Pipeline:
    '''Une características + modelo en un solo objeto: recibe datos crudos y predice.'''
    return Pipeline([
        ('features', crear_pipeline_features(escalar=escalar, con_puntaje=con_puntaje)),
        ('modelo', clasificador),
    ])


# ---------------------------------------------------------------------------
# 2. MÉTRICAS (clase de interés = 0, 'no pagó')
# ---------------------------------------------------------------------------
def proba_no_pago(pipeline, X) -> np.ndarray:
    '''Probabilidad de la clase 0 ('no pagó'). Se busca por classes_, no por posición fija.'''
    indice = list(pipeline.classes_).index(0)
    return pipeline.predict_proba(X)[:, indice]


def calcular_metricas(y_true, p_no_pago, umbral: float = UMBRAL_DEFECTO) -> dict:
    '''
    Métricas de un modelo. Se predice 'no pagó' cuando la probabilidad de no pago >= umbral.
    '''
    es_no_pago = (np.asarray(y_true) == 0).astype(int)     # 1 = evento de interés
    pred_no_pago = (p_no_pago >= umbral).astype(int)
    return {
        'accuracy': accuracy_score(es_no_pago, pred_no_pago),
        'precision': precision_score(es_no_pago, pred_no_pago, zero_division=0),
        'recall': recall_score(es_no_pago, pred_no_pago, zero_division=0),
        'f1': f1_score(es_no_pago, pred_no_pago, zero_division=0),
        'roc_auc': roc_auc_score(es_no_pago, p_no_pago),
        'pr_auc': average_precision_score(es_no_pago, p_no_pago),
        'no_pago_predichos': int(pred_no_pago.sum()),
        'no_pago_reales': int(es_no_pago.sum()),
    }


def _pr_auc_no_pago(estimador, X, y):
    '''Scorer para la validación cruzada: PR-AUC de la clase 'no pagó'.'''
    return average_precision_score((np.asarray(y) == 0).astype(int), proba_no_pago(estimador, X))


# Scorers de validación cruzada (todos sobre la clase 0)
SCORERS_CV = {
    'roc_auc': 'roc_auc',   # simétrico: es el mismo valor para la clase 0 o la 1
    'pr_auc': _pr_auc_no_pago,
    'precision': make_scorer(precision_score, pos_label=0, zero_division=0),
    'recall': make_scorer(recall_score, pos_label=0, zero_division=0),
    'f1': make_scorer(f1_score, pos_label=0, zero_division=0),
}


def validacion_cruzada(pipeline, X_train, y_train, n_folds: int = N_FOLDS) -> dict:
    '''
    Validación cruzada ESTRATIFICADA solo sobre el conjunto de entrenamiento.
    Devuelve media y desvío de cada métrica. En cada fold el pipeline completo (preprocesador
    incluido) se ajusta únicamente con las filas de entrenamiento de ese fold.
    '''
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEMILLA)
    res = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=SCORERS_CV, n_jobs=1)
    salida = {}
    for metrica in SCORERS_CV:
        valores = res[f'test_{metrica}']
        salida[f'cv_{metrica}_media'] = valores.mean()
        salida[f'cv_{metrica}_std'] = valores.std()
    return salida


# ---------------------------------------------------------------------------
# 3. ENTRENAMIENTO Y COMPARACIÓN
# ---------------------------------------------------------------------------
def comparar_modelos(datos: dict, con_puntaje: bool = False, umbral: float = UMBRAL_DEFECTO,
                     con_cv: bool = True) -> dict:
    '''
    Entrena todos los modelos con train y los evalúa en train, test y validación cruzada.

    Devuelve:
        'metricas'  : DataFrame (modelo, conjunto, métricas) con filas train y test
        'cv'        : DataFrame con media y desvío de la validación cruzada por modelo
        'pipelines' : {nombre: pipeline ajustado}
        'proba_test': {nombre: probabilidad de no pago en test} (para curvas y matrices)
    '''
    X_train, X_test = datos['X_train'], datos['X_test']
    y_train, y_test = datos['y_train'], datos['y_test']

    filas, filas_cv, pipelines, proba_test = [], [], {}, {}
    for nombre, (clasificador, escalar) in crear_modelos(y_train).items():
        pipe = construir_pipeline(clasificador, escalar, con_puntaje)

        inicio = time.time()
        pipe.fit(X_train, y_train)
        tiempo = time.time() - inicio

        for conjunto, X, y in (('train', X_train, y_train), ('test', X_test, y_test)):
            p = proba_no_pago(pipe, X)
            fila = {'modelo': nombre, 'conjunto': conjunto, 'tiempo_seg': round(tiempo, 2)}
            fila.update(calcular_metricas(y, p, umbral))
            filas.append(fila)
            if conjunto == 'test':
                proba_test[nombre] = p

        if con_cv:
            fila_cv = {'modelo': nombre}
            fila_cv.update(validacion_cruzada(pipe, X_train, y_train))
            filas_cv.append(fila_cv)
        pipelines[nombre] = pipe

    return {
        'metricas': pd.DataFrame(filas),
        'cv': pd.DataFrame(filas_cv).set_index('modelo') if con_cv else None,
        'pipelines': pipelines,
        'proba_test': proba_test,
    }


# ---------------------------------------------------------------------------
# 4. SELECCIÓN DEL MODELO Y DEL UMBRAL
# ---------------------------------------------------------------------------
def elegir_modelo(tabla_cv: pd.DataFrame) -> str:
    '''
    Criterio de selección: mayor PR-AUC medio en validación cruzada (se excluye la referencia).
    Se usa la validación cruzada y no el test porque el test tiene solo ~100 no pagos: elegir
    con él mezclaría la selección con la evaluación final.
    '''
    candidatos = tabla_cv.drop(index='Referencia (prior)', errors='ignore')
    return candidatos['cv_pr_auc_media'].idxmax()


def umbral_optimo_f1(pipeline, X_train, y_train, n_folds: int = N_FOLDS) -> tuple:
    '''
    Busca el umbral de probabilidad de no pago que maximiza el F1 de la clase 'no pagó'.
    Se calcula con predicciones fuera de fold (out-of-fold) sobre TRAIN: cada fila se predice
    con un modelo que no la vio. Así el umbral no se ajusta con el test.
    Devuelve (umbral, f1_en_oof).
    '''
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEMILLA)
    oof = cross_val_predict(pipeline, X_train, y_train, cv=cv, method='predict_proba', n_jobs=1)
    p_no_pago = oof[:, list(np.sort(np.unique(y_train))).index(0)]
    es_no_pago = (np.asarray(y_train) == 0).astype(int)

    precision, recall, umbrales = precision_recall_curve(es_no_pago, p_no_pago)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    mejor = int(np.argmax(f1))
    return float(umbrales[mejor]), float(f1[mejor])


def guardar_modelo(pipeline, umbral: float, nombre: str, metricas_test: dict,
                   ruta: Path = RUTA_MODELO) -> Path:
    '''
    Guarda el pipeline ajustado (preprocesamiento + modelo) junto con su metadata.
    La API (Avance 4) carga este único archivo: no repite ninguna transformación a mano.
    '''
    paquete = {
        'pipeline': pipeline,
        'umbral_no_pago': umbral,
        'nombre_modelo': nombre,
        'columnas_entrada': columnas_entrada(con_puntaje=False),
        'metricas_test': metricas_test,
        'semilla': SEMILLA,
    }
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)   # crea models/ si todavía no existe
    joblib.dump(paquete, ruta, compress=3)
    return ruta


# ---------------------------------------------------------------------------
# 5. GRÁFICO DE COMPARACIÓN
# ---------------------------------------------------------------------------
def graficar_comparacion(tabla_test: pd.DataFrame, titulo: str = 'Comparación de modelos (test)'):
    '''
    Barras horizontales, una por métrica; en el color de acento va el mejor modelo de cada panel.
    `tabla_test` tiene una fila por modelo (índice = nombre) y una columna por métrica.
    '''
    metricas = [('roc_auc', 'ROC-AUC'), ('pr_auc', 'PR-AUC'), ('precision', 'Precision (no pagó)'),
                ('recall', 'Recall (no pagó)'), ('f1', 'F1 (no pagó)'), ('accuracy', 'Accuracy (solo referencia)')]
    fig, ejes = plt.subplots(2, 3, figsize=(16, 8))
    for eje, (col, etiqueta) in zip(ejes.flatten(), metricas):
        datos = tabla_test[col].sort_values()
        colores = ['#C0392B' if m == datos.idxmax() else '#2E86AB' for m in datos.index]
        eje.barh(datos.index, datos.values, color=colores)
        eje.set_title(etiqueta, fontweight='bold')
        eje.tick_params(axis='y', labelsize=8)
        for i, v in enumerate(datos.values):
            eje.text(v, i, f' {v:.3f}', va='center', fontsize=8)
    fig.suptitle(titulo, fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. EJECUCIÓN COMPLETA
# ---------------------------------------------------------------------------
def main():
    '''Entrena, compara, elige el mejor modelo sin `puntaje`, define el umbral y lo guarda.'''
    datos = ft_engineering(con_puntaje=False)
    resultado = comparar_modelos(datos, con_puntaje=False)

    nombre = elegir_modelo(resultado['cv'])
    pipe = resultado['pipelines'][nombre]
    umbral, f1_oof = umbral_optimo_f1(pipe, datos['X_train'], datos['y_train'])
    metricas_test = calcular_metricas(datos['y_test'], proba_no_pago(pipe, datos['X_test']), umbral)

    ruta = guardar_modelo(pipe, umbral, nombre, metricas_test)
    print(resultado['cv'].round(4).to_string())
    print(f'\nModelo elegido: {nombre} | umbral (F1 out-of-fold = {f1_oof:.3f}): {umbral:.3f}')
    print(f'Métricas en test con ese umbral: { {k: round(v, 4) for k, v in metricas_test.items()} }')
    print(f'Guardado en: {ruta}')


if __name__ == '__main__':
    main()
