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

## Ejecutar tests con pytest (modo simple)
test:
	@echo "[Make] Ejecutando tests..."
	pytest -q

## Ejecutar tests con más detalle
test-verbose:
	@echo "[Make] Ejecutando tests con detalle..."
	pytest -vv --tb=short

## Ejecutar tests específicos
test-file:
	@echo "[Make] Ejecutando test específico: $(TEST_FILE)"
	pytest $(TEST_FILE) -vv

## Ejecutar tests con cobertura
test-coverage:
	@echo "[Make] Ejecutando tests con cobertura..."
	coverage run -m pytest -q
	coverage report -m

## Generar reporte HTML de cobertura
coverage-html:
	@echo "[Make] Generando reporte HTML de cobertura..."
	coverage run -m pytest -q
	coverage html
	@echo "Abre htmlcov/index.html en tu navegador"

## Ejecutar tests y abrir reporte de cobertura
coverage: coverage-html
	@echo "[Make] Abriendo reporte de cobertura..."
	@cmd /c start htmlcov/index.html 2>nul || open htmlcov/index.html 2>/dev/null || xdg-open htmlcov/index.html 2>/dev/null

## Ejecutar tests rápidos (solo unit tests, sin integration)
test-fast:
	@echo "[Make] Ejecutando tests rápidos..."
	pytest tests/test_counter.py tests/test_detector_mapping.py tests/test_utils.py -q

## Ejecutar tests de integración
test-integration:
	@echo "[Make] Ejecutando tests de integración..."
	pytest tests/test_headless_integration.py -v

## Verificar que todos los tests pasan antes de commit
test-all: clean-test
	@echo "[Make] Ejecutando suite completa de tests..."
	pytest tests/ -v --tb=short
	@echo "[Make] Verificando cobertura mínima..."
	coverage run -m pytest tests/ -q
	coverage report --fail-under=35

## Formatear código con Black
format:
	@echo "[Make] Formateando código..."
	black src tests

## Limpiar archivos temporales y reportes
clean:
	@echo "[Make] Limpiando..."
	-rm -rf __pycache__ .pytest_cache .mypy_cache
	-rm -rf $(REPORTS_DIR)/*.csv

## Limpiar archivos de tests y cobertura
clean-test:
	@echo "[Make] Limpiando archivos de test..."
	-rm -rf .pytest_cache
	-rm -rf htmlcov
	-rm -f .coverage
	-rm -f coverage.xml

## Ayuda: lista todas las reglas
help:
	@echo "====================================="
	@echo "Comandos disponibles:"
	@echo "====================================="
	@echo ""
	@echo "EJECUCIÓN:"
	@grep -E '^## ' Makefile | grep -E '(run-|serve|grpc)' | sed -e 's/## /  /'
	@echo ""
	@echo "TESTING:"
	@grep -E '^## ' Makefile | grep -E '(test|coverage)' | sed -e 's/## /  /'
	@echo ""
	@echo "DOCKER:"
	@grep -E '^## ' Makefile | grep -E 'docker' | sed -e 's/## /  /'
	@echo ""
	@echo "UTILIDADES:"
	@grep -E '^## ' Makefile | grep -E '(format|clean|help)' | sed -e 's/## /  /'
	@echo ""
	@echo "====================================="

