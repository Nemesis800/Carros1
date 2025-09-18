# ==========================================================
# Makefile - Proyecto de Detección y Conteo de Vehículos
# ==========================================================

# Variables configurables
PYTHON       ?= python
DOCKER_IMAGE ?= contador-vehiculos:latest
SRC_VIDEO    ?= videos/prueba1.MP4
MODEL        ?= yolo11n.pt
CSV_NAME     ?= reporte
CONF         ?= 0.30
ORIENT       ?= vertical
LINE_POS     ?= 0.50
CAP_CAR      ?= 80
CAP_MOTO     ?= 60

# Directorios
REPORTS_DIR  := reports
CACHE_DIR    := .ultralytics_cache

# ==========================================================
# Reglas principales
# ==========================================================

## Ejecutar en modo CLI (sin UI)
run-cli:
	@echo "[Make] Ejecutando en modo CLI..."
	$(PYTHON) src/app.py --cli \
		--source "$(or $(SRC),$(SRC_VIDEO))" \
		--model $(MODEL) \
		--conf $(CONF) \
		--orientation $(ORIENT) \
		--line-pos $(LINE_POS) \
		--cap-car $(CAP_CAR) \
		--cap-moto $(CAP_MOTO) \
		--csv-name $(CSV_NAME) \
		--csv-dir $(REPORTS_DIR) \
		--no-display

## Ejecutar en modo UI (Tkinter)
run-ui:
	@echo "[Make] Ejecutando en modo UI..."
	$(PYTHON) src/app.py

## Lanzar servidor gRPC
serve:
	@echo "[Make] Ejecutando servidor gRPC..."
	$(PYTHON) services/inference_server.py

## Cliente gRPC (requiere un video como argumento)
grpc-client:
	@echo "[Make] Cliente gRPC..."
	$(PYTHON) clients/grpc_client.py "$(or $(SRC),$(SRC_VIDEO))"

# ==========================================================
# Docker
# ==========================================================

## Construir imagen Docker
docker-build:
	@echo "[Make] Construyendo imagen Docker..."
	docker build -t $(DOCKER_IMAGE) .

## Ejecutar CLI dentro de contenedor Docker
docker-run-cli:
	@if not exist "$(REPORTS_DIR)" mkdir "$(REPORTS_DIR)"
	@if not exist "$(CACHE_DIR)" mkdir "$(CACHE_DIR)"
	docker run --rm -it \
		-v "$(CURDIR)/$(REPORTS_DIR)":/app/reports \
		-v "$(CURDIR)/$(CACHE_DIR)":/root/.cache/ultralytics \
		-v "$(SRC)":/data/input.mp4:ro \
		$(DOCKER_IMAGE) \
		python src/app.py --cli \
			--source /data/input.mp4 \
			--model $(MODEL) \
			--conf $(CONF) \
			--orientation $(ORIENT) \
			--line-pos $(LINE_POS) \
			--cap-car $(CAP_CAR) \
			--cap-moto $(CAP_MOTO) \
			--csv-name $(CSV_NAME) \
			--csv-dir /app/reports \
			--no-display

# ==========================================================
# Utilidades
# ==========================================================

## Ejecutar tests con pytest
test:
	@echo "[Make] Ejecutando tests..."
	pytest -q

## Formatear código con Black
format:
	@echo "[Make] Formateando código..."
	black src tests

## Limpiar archivos temporales y reportes
clean:
	@echo "[Make] Limpiando..."
	-rm -rf __pycache__ .pytest_cache .mypy_cache
	-rm -rf $(REPORTS_DIR)/*.csv

## Ayuda: lista todas las reglas
help:
	@echo "Comandos disponibles:"
	@grep -E '^##' Makefile | sed -e 's/## //'

