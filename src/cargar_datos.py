import os
import pandas as pd


def cargarDatos():
    # Obtiene la ruta del directorio donde se encuentra este script (mlops_pipeline/src/)
    script_dir = os.path.dirname(__file__)
    # Sube dos niveles hasta la raíz del repo, donde está Base_de_datos.csv
    file_path = os.path.join(script_dir, '..', 'Base_de_datos.csv')
    # Lee el archivo CSV usando la ruta completa
    df = pd.read_csv(file_path, parse_dates=['fecha_prestamo'])
    return df


if __name__ == '__main__':
    df = cargarDatos()
    print(df.shape)
