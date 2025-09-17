# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Ultralytics/torch usan ~/.cache; dejamos una ruta clara para volumen
    ULT_CACHE_DIR=/root/.cache/ultralytics

WORKDIR /app

# Dependencias del sistema para OpenCV/ffmpeg y compatibilidad headless
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Instala deps primero para cache de capas
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# (Opcional) instala deps de test si vas a correr pytest dentro del contenedor
# COPY requirements-dev.txt /app/requirements-dev.txt
# RUN pip install -r requirements-dev.txt

# Copia el resto del código
COPY src/ /app/src/
COPY README.md CASOS_USO.md CONTRIBUTING.md /app/

# Crea carpeta de reportes por defecto
RUN mkdir -p /app/reports

# Usuario no-root (opcional)
RUN useradd -ms /bin/bash appuser && chown -R appuser:appuser /app /root
USER appuser

# Comando por defecto: mostrar ayuda CLI (para ver flags disponibles)
CMD ["python", "src/app.py", "--cli", "--help"]
