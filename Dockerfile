# ============================================================================
# Imagen Docker de la API de riesgo crediticio (PIM5 · Avance 4)
# Construir:  docker build -t pim5-api .
# Ejecutar:   docker run -d -p 8000:8000 --name pim5-api pim5-api
# Probar:     http://localhost:8000/docs
# ============================================================================

# 1. Imagen base: la misma versión de Python con la que se entrenó el modelo (3.13)
FROM python:3.13-slim

# 2. Configuración de Python dentro del contenedor:
#    no generar archivos .pyc y mostrar los logs al instante
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 3. LightGBM necesita la librería del sistema libgomp (no viene en la imagen slim)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 4. Carpeta de trabajo dentro del contenedor
WORKDIR /app

# 5. Dependencias primero: Docker las guarda en caché y no las reinstala si solo cambia el código.
#    Se instala el requirements.txt completo del proyecto, como pide la consigna.
#    (Alternativa más liviana: un archivo solo con las librerías de la API.)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Código fuente y modelo entrenado
COPY src/ ./src/
COPY models/ ./models/

# 7. Usuario sin privilegios de administrador (buena práctica de seguridad)
RUN useradd --create-home appuser
USER appuser

# 8. Puerto de la API
EXPOSE 8000

# 9. Chequeo de salud: Docker marca el contenedor como "healthy" si /health responde
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# 10. Servidor de aplicación: Uvicorn sirve la app de FastAPI definida en src/model_deploy.py
CMD ["uvicorn", "src.model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]
