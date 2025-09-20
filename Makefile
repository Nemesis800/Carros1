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
	uv run -p .venv python src/app.py --cli \
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
	uv run -p .venv python src/app.py

## Lanzar servidor gRPC
serve:
	@echo "[Make] Ejecutando servidor gRPC..."
	uv run -p .venv python services/inference_server.py

## Cliente gRPC (requiere un video como argumento)
grpc-client:
	@echo "[Make] Cliente gRPC..."
	uv run -p .venv python clients/grpc_client.py "$(or $(SRC),$(SRC_VIDEO))"

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
	uv run -p .venv pytest -q

## Ejecutar tests con más detalle
test-verbose:
	@echo "[Make] Ejecutando tests con detalle..."
	uv run -p .venv pytest -vv --tb=short

## Ejecutar tests específicos
test-file:
	@echo "[Make] Ejecutando test específico: $(TEST_FILE)"
	uv run -p .venv pytest $(TEST_FILE) -vv

## Ejecutar tests con cobertura
test-coverage:
	@echo "[Make] Ejecutando tests con cobertura..."
	uv run -p .venv coverage run -m pytest -q
	uv run -p .venv coverage report -m

## Generar reporte HTML de cobertura
coverage-html:
	@echo "[Make] Generando reporte HTML de cobertura..."
	uv run -p .venv coverage run -m pytest -q
	uv run -p .venv coverage html
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
	uv run -p .venv black src tests

## Limpiar archivos temporales y reportes
clean:
	@echo "[Make] Limpiando..."
	-if exist __pycache__ rd /s /q __pycache__
	-if exist .pytest_cache rd /s /q .pytest_cache
	-if exist .mypy_cache rd /s /q .mypy_cache
	-if exist "$(REPORTS_DIR)\*.csv" del /q "$(REPORTS_DIR)\*.csv"

## Limpiar archivos de tests y cobertura
clean-test:
	@echo "[Make] Limpiando archivos de test..."
	-if exist .pytest_cache rd /s /q .pytest_cache
	-if exist htmlcov rd /s /q htmlcov
	-if exist .coverage del /q .coverage
	-if exist coverage.xml del /q coverage.xml

## Ayuda: lista todas las reglas
help:
	@echo =====================================
	@echo Comandos disponibles:
	@echo =====================================
	@echo.
	@echo EJECUCION:
	@echo   run-ui         - Ejecutar en modo UI (Tkinter)
	@echo   run-cli        - Ejecutar en modo CLI (sin UI)
	@echo   serve          - Lanzar servidor gRPC
	@echo   grpc-client    - Cliente gRPC (requiere un video como argumento)
	@echo.
	@echo TESTING:
	@echo   test          - Ejecutar tests con pytest (modo simple)
	@echo   test-verbose  - Ejecutar tests con mas detalle
	@echo   test-file     - Ejecutar tests especificos
	@echo   test-coverage - Ejecutar tests con cobertura
	@echo   coverage-html - Generar reporte HTML de cobertura
	@echo   test-fast     - Ejecutar tests rapidos
	@echo   test-integration - Ejecutar tests de integracion
	@echo   test-all      - Verificar que todos los tests pasan
	@echo.
	@echo DOCKER:
	@echo   docker-build  - Construir imagen Docker
	@echo   docker-run-cli - Ejecutar CLI dentro de contenedor Docker
	@echo.
	@echo UTILIDADES:
	@echo   format       - Formatear codigo con Black
	@echo   clean        - Limpiar archivos temporales y reportes
	@echo   clean-test   - Limpiar archivos de tests y cobertura
	@echo.
	@echo =====================================

