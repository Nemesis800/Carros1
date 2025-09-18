# ==========================================================
# Dockerfile para el sistema de conteo de vehículos
# ==========================================================

# --- Etapa base: Python slim + dependencias del sistema ---
FROM python:3.10-slim AS base

# Evitar prompts interactivos de apt
ENV DEBIAN_FRONTEND=noninteractive

# Instalamos dependencias del sistema necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Copiar requerimientos primero (para aprovechar cache de Docker)
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# --- Etapa final ---
FROM base AS final

# Copiar código fuente y otros archivos necesarios
COPY src/ src/
COPY proto/ proto/
COPY clients/ clients/
COPY services/ services/

# Directorio para reportes CSV
RUN mkdir -p /app/reports

# Variables de entorno por defecto
ENV PYTHONUNBUFFERED=1 \
    CSV_DIR=/app/reports \
    MODEL_NAME=yolo11n.pt

# Puerto gRPC (50051) y posible UI web (8080)
EXPOSE 50051 8080

# Comando por defecto: servidor gRPC
CMD ["python", "services/inference_server.py"]
