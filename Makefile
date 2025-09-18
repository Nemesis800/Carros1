# ====== Config ======
IMAGE = contador-vehiculos
TAG   = latest
PY    = python

# Parámetros por defecto para CLI
MODEL     ?= yolo12n.pt
CONF      ?= 0.30
ORIENT    ?= vertical
LINE_POS  ?= 0.50
CAP_CAR   ?= 50
CAP_MOTO  ?= 50
CSV_DIR   ?= reports
CSV_NAME  ?=
SRC       ?=
WEBCAM    ?=

# ====== Ayuda ======
help:
	@echo Targets disponibles:
	@echo   make venv             - crear venv local (.venv)
	@echo   make install          - instalar deps (requirements.txt)
	@echo   make install-dev      - instalar deps de dev (pytest/coverage)
	@echo   make run-ui           - ejecutar app con interfaz (Tkinter)
	@echo   make run-cli SRC=...  - ejecutar app en CLI headless (usar SRC="C:/ruta/video.mp4" o WEBCAM=1)
	@echo   make test             - ejecutar pytest
	@echo   make cov              - coverage + reporte html
	@echo   make docker-build     - construir imagen Docker
	@echo   make docker-run-cli   - ejecutar en Docker (CLI headless) con SRC=... o WEBCAM=1

# ====== Venv & deps (Windows cmd) ======
venv:
	$(PY) -m venv .venv

install:
	.venv\Scripts\activate && pip install -r requirements.txt

install-dev:
	.venv\Scripts\activate && pip install -r requirements.txt && pip install -r requirements-dev.txt

# ====== Run local (sin Docker) ======
run-ui:
	.venv\Scripts\activate && python src\app.py

# Si WEBCAM=1 -> usa webcam; en otro caso usa SRC (ruta al video).
run-cli:
ifeq ($(strip $(WEBCAM)),1)
	.venv\Scripts\activate && python src\app.py --cli --webcam --model $(MODEL) --conf $(CONF) --orientation $(ORIENT) --line-pos $(LINE_POS) --cap-car $(CAP_CAR) --cap-moto $(CAP_MOTO) --csv --csv-dir "$(CSV_DIR)" --csv-name "$(CSV_NAME)" --no-display
else
	.venv\Scripts\activate && python src\app.py --cli --source "$(SRC)" --model $(MODEL) --conf $(CONF) --orientation $(ORIENT) --line-pos $(LINE_POS) --cap-car $(CAP_CAR) --cap-moto $(CAP_MOTO) --csv --csv-dir "$(CSV_DIR)" --csv-name "$(CSV_NAME)" --no-display
endif


# ====== Tests ======
test:
	.venv\Scripts\activate && pytest -q

cov:
	.venv\Scripts\activate && coverage run -m pytest -q && coverage report -m && coverage html

# ====== Docker ======
docker-build:
	docker build -t contador-vehiculos:latest .

# Si WEBCAM=1 -> webcam; si no, monta el archivo de video indicado en SRC
docker-run-cli:
	@if not exist "reports" mkdir reports
	@if not exist ".ultralytics_cache" mkdir .ultralytics_cache
ifeq ($(strip $(WEBCAM)),1)
	docker run --rm -it -v "$(CURDIR)\reports:/app/reports" -v "$(CURDIR)\.ultralytics_cache:/root/.cache/ultralytics" contador-vehiculos:latest python src/app.py --cli --webcam --model $(MODEL) --conf $(CONF) --orientation $(ORIENT) --line-pos $(LINE_POS) --cap-car $(CAP_CAR) --cap-moto $(CAP_MOTO) --csv --csv-dir /app/reports --csv-name "$(CSV_NAME)" --no-display
else
	docker run --rm -it -v "$(CURDIR)\reports:/app/reports" -v "$(CURDIR)\.ultralytics_cache:/root/.cache/ultralytics" -v "$(SRC):/data/input.mp4:ro" contador-vehiculos:latest python src/app.py --cli --source /data/input.mp4 --model $(MODEL) --conf $(CONF) --orientation $(ORIENT) --line-pos $(LINE_POS) --cap-car $(CAP_CAR) --cap-moto $(CAP_MOTO) --csv --csv-dir /app/reports --csv-name "$(CSV_NAME)" --no-display
endif

run-ui-streamlit:
\tuv run -p .venv streamlit run src/ui_streamlit/app.py

docker-ui:
\tdocker build -t $(USER)/contador-ui:latest .
\tdocker run --rm -p 8501:8501 -v "$(PWD)/reports:/app/reports" $(USER)/contador-ui:latest


