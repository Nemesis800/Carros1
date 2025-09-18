# Detección, Seguimiento y Conteo de Vehículos en Tiempo Real (YOLOv11 + Supervision + MLflow)

Este proyecto detecta, sigue y cuenta vehículos (carros y motos) en tiempo real a partir de un video cargado manualmente o la webcam. Mantiene un inventario por tipo de vehículo con capacidades configurables y genera una alarma visual y auditiva cuando se excede la capacidad definida para cada tipo.  

Incluye:
- **YOLOv8, YOLOv11 y YOLOv12** como modelos de detección.
- **🆕 Integración completa con MLflow** para seguimiento de experimentos y métricas.
- **Reportes CSV configurables** con registro automático en MLflow.
- **Modo CLI headless** y **UI (Tkinter / Streamlit)**.
- **Pruebas automáticas con pytest** y **coverage**.
- **Servicios separados vía gRPC** (inferencia y UI/cliente).
- Compatibilidad con **Docker** y automatización vía **Makefile**.

---

## 🚀 Tecnologías principales
- **Ultralytics YOLO v8/v11/v12** – detección de objetos
- **🆕 MLflow** – seguimiento de experimentos, métricas y gestión de modelos
- **Supervision (ByteTrack)** – seguimiento y anotación
- **OpenCV** – lectura de video y visualización optimizada
- **Tkinter / Streamlit** – interfaces gráficas
- **Matplotlib / Seaborn** – visualizaciones automáticas
- **psutil** – información del sistema
- **Pytest / Coverage** – pruebas y reportes
- **gRPC / Protobuf** – comunicación entre servicios
- **Docker / Makefile** – despliegue y automatización

---

## ✨ Características
- Detección y conteo de vehículos en tiempo real.
- Seguimiento multi-objeto con IDs únicos.
- **🗺️ Interfaz optimizada**: Ventana única OpenCV sin ventanas duplicadas.
- Inventario dinámico con alarmas visuales/sonoras.
- **🆕 Seguimiento automático de experimentos con MLflow**:
  - Registro de parámetros de configuración
  - Métricas en tiempo real (FPS, detecciones, conteos)
  - Gestión de artefactos (CSV, modelos)
  - **Model Registry** con versionado automático
  - **Visualizaciones automáticas** de métricas
  - **Información del sistema** (hardware, software)
  - Interfaz web para visualización de experimentos
- Exportación de reportes CSV (IN/OUT y SUMMARY) con registro en MLflow.
- Modo CLI headless para entornos sin interfaz gráfica.
- Interfaz gráfica con Tkinter y **Streamlit (cliente web)**.
- Servicios desacoplados:
  - `inference_server` (procesamiento YOLO).
  - `streamlit_app` (UI web que consume vía gRPC).
- Pruebas unitarias e integración con pytest.
- Dockerfile listo para build/run.

---

## 💻 Requisitos
- Python **3.11+**
- [UV](https://github.com/astral-sh/uv) o pip para dependencias
- Windows, Linux o macOS
- Webcam opcional
- GPU NVIDIA opcional (CUDA/cuDNN)

---

## 📦 Instalación
```bash
# Crear entorno virtual
uv venv .venv
.\.venv\Scripts\activate

# Instalar dependencias
uv pip install -r requirements.txt      # Incluye MLflow automáticamente
uv pip install -r requirements-dev.txt   # dependencias de desarrollo
```

---

## ▶️ Ejecución con interfaz Tkinter
```powershell
# Con UV
uv run -p .venv python src/app.py

# Con pip tradicional
.\.venv\Scripts\activate.bat
python src/app.py
```

---

## 🌐 Ejecución con interfaz Streamlit
```powershell
streamlit run streamlit_app.py
```
Esto abre una página web en http://localhost:8501 donde puedes:
- Subir un video o usar webcam.
- Configurar modelo, confianza, línea de conteo y capacidades.
- Ver los frames procesados en tiempo real.
- Descargar el CSV generado.
- **🆕 Todo automáticamente registrado en MLflow**.

---

## ⚡ Modo CLI (headless)
Ejemplo con webcam:
```powershell
python src/app.py --cli --webcam --model yolo12n.pt --conf 0.30 ^
  --orientation vertical --line-pos 0.50 ^
  --cap-car 50 --cap-moto 50 ^
  --csv --csv-dir reports --csv-name "turno_noche" --no-display
```

Ejemplo con archivo de video:
```powershell
python src/app.py --cli --source "C:\Videos\ejemplo.mp4" --model yolo11n.pt ^
  --orientation horizontal --line-pos 0.25 --invert ^
  --csv --csv-dir "C:\Users\CAMILO\Desktop\reports" --csv-name "parqueadero_sabado" --no-display
```

---

## 🧰 Uso de Makefile
El proyecto incluye un `Makefile` completo para simplificar tareas comunes:

```bash
# ===== EJECUCIÓN =====
# Ejecutar en modo CLI con video de ejemplo
make run-cli SRC="videos/prueba1.MP4" MODEL=yolo12n.pt CONF=0.3 ORIENT=vertical LINE_POS=0.5

# Ejecutar con interfaz Tkinter
make run-ui

# Levantar servidor gRPC
make serve

# Cliente gRPC
make grpc-client SRC="videos/prueba1.MP4"

# ===== TESTING (NUEVO) =====
# Todos los tests (49 tests)
make test

# Tests detallados
make test-verbose

# Tests específicos
make test-file TEST_FILE=tests/test_counter.py

# Tests con cobertura
make test-coverage

# Reporte HTML de cobertura (abre automáticamente)
make coverage

# Tests rápidos (solo unit tests)
make test-fast

# Suite completa con validación
make test-all

# ===== DOCKER =====
# Ejecutar en Docker en modo CLI
make docker-run-cli SRC="videos/prueba1.MP4"

# Construir imagen Docker
make docker-build

# ===== UTILIDADES =====
# Formatear código
make format

# Limpiar archivos temporales
make clean

# Limpiar archivos de tests
make clean-test

# Ver todos los comandos disponibles
make help
```

---

## 🐳 Docker
Construir imagen:
```bash
docker build -t contador-vehiculos .
```

Ejecutar contenedor (procesar un video):
```bash
docker run --rm -it ^
  -v "%cd%\reports:/app/reports" ^
  -v "%cd%\videos\prueba1.MP4:/data/input.mp4:ro" ^
  contador-vehiculos python src/app.py --cli --source /data/input.mp4 --no-display
```

---

## 📄 Reporte CSV
Columnas:
```
timestamp;evento;clase;car_in;car_out;moto_in;moto_out;car_inv;moto_inv;modelo;conf;orientacion;pos_linea;invertido;fuente
```

Ejemplo:
```
2025-09-17T12:34:56;IN;car;1;0;0;0;1;0;yolo12n.pt;0.30;vertical;0.50;False;ejemplo.mp4
2025-09-17T12:40:00;SUMMARY;-;15;10;4;5;5;-1;yolo12n.pt;0.30;vertical;0.50;False;ejemplo.mp4
```

---

## 🧪 Sistema de Pruebas Completo

### ✅ **Suite de Testing Robusta: 49 Tests (100% Pasando)**

**📊 Cobertura por Módulo:**
```
📈 Cobertura Total: 55%

🏆 Módulos con Excelente Cobertura:
├── CLI (cli.py): 92% ✅
├── Utils (utils.py): 90% ✅  
├── Counter (counter.py): 85% ✅
├── gRPC Client: 79% ✅
└── MLflow Integration: 77% ✅

🔶 Módulos con Buena Cobertura:
├── Detector (detector.py): 65%
├── Processor (processor.py): 61%
└── gRPC Server: 43%

📝 Módulos Funcionales (lógica testeada):
├── UI Tkinter: Tests de lógica de negocio
└── UI Streamlit: Tests de componentes web
```

**🗂️ Tipos de Tests Implementados:**
- ✅ **Unit Tests** (30 tests): Funciones individuales
- ✅ **Integration Tests** (12 tests): Pipeline completo
- ✅ **UI Logic Tests** (7 tests): Lógica sin dependencias de ventana

### 🚀 **Comandos de Testing Mejorados**

```powershell
# ===== BÁSICOS =====
# Ejecutar todos los tests (49 tests)
make test

# Tests con más detalle
make test-verbose

# Tests específicos
make test-file TEST_FILE=tests/test_counter.py

# ===== COBERTURA =====
# Ver cobertura en consola
make test-coverage

# Generar reporte HTML interactivo
make coverage
# Abre automáticamente htmlcov/index.html

# Solo generar HTML sin abrir
make coverage-html

# ===== ESPECIALIZADOS =====
# Tests rápidos (solo unit tests)
make test-fast

# Tests de integración
make test-integration

# Verificación completa antes de commit
make test-all

# Limpiar archivos de test
make clean-test
```

### 📋 **Tests por Módulo**

**🔧 Core Logic:**
- `test_counter.py` (4 tests): Lógica de conteo y cruce de líneas
- `test_detector_mapping.py` (1 test): Mapeo de clases YOLO
- `test_processor.py` (via integration): Pipeline de procesamiento

**⚙️ Interfaces:**
- `test_cli.py` (7 tests): Parsing de argumentos y ejecución CLI
- `test_ui_app.py` (8 tests): Lógica de Tkinter (sin ventanas)
- `test_streamlit_app.py` (10 tests): Componentes web (sin servidor)

**🌐 Servicios:**
- `test_grpc_services.py` (6 tests): Cliente y servidor gRPC
- `test_mlflow_integration.py` (7 tests): Tracking de experimentos

**🔨 Utilidades:**
- `test_utils.py` (4 tests): Funciones auxiliares multiplataforma
- `test_app_csv.py` (1 test): Generación de reportes CSV
- `test_headless_integration.py` (1 test): Pipeline headless completo

### 🎯 **Configuración de Desarrollo**

```powershell
# Instalar dependencias de desarrollo
uv pip install -r requirements-dev.txt

# Configuración incluye:
# ├── pytest: Framework de testing
# ├── coverage: Análisis de cobertura
# ├── black: Formateo de código
# ├── ruff: Linting rápido
# ├── pre-commit: Hooks de git
# └── mypy-extensions: Type checking
```

### 📊 **Análisis de Cobertura Detallado**

```bash
# Ver líneas específicas sin cobertura
coverage report -m

# Generar reporte XML (para CI/CD)
coverage xml

# Reporte con skip de líneas ya cubiertas
coverage report --skip-covered

# Verificar cobertura mínima (falla si < 35%)
coverage report --fail-under=35
```

**🔍 Analiza el reporte HTML en `htmlcov/index.html` para:**
- Líneas exactas sin cobertura (en rojo)
- Branches no ejecutados (en amarillo)
- Funciones completamente testeadas (en verde)
- Métricas detalladas por archivo

---

## 🔌 Arquitectura gRPC
El sistema se divide en dos servicios:

```
┌───────────────┐     gRPC (protobuf)     ┌──────────────────┐
│   Cliente      │ <────────────────────> │   Servidor        │
│ (Streamlit o  │                        │ (inference_server)│
│  grpc_client) │                        │                  │
└───────────────┘                        └──────────────────┘
         ▲                                         │
         │                                         ▼
   Usuario/Front                                YOLO + CSV
```
## 🏗️ Arquitectura del Software

El sistema está organizado en módulos dentro de la carpeta `src/`.  
Cada uno cumple un rol específico y se integran a través del **VideoProcessor**.

```text
                         ┌───────────────────────────┐
                         │        app.py             │
                         │ Punto de entrada          │
                         │ - Decide CLI o UI         │
                         └───────────┬───────────────┘
                                     │
           ┌─────────────────────────┴───────────────────────────┐
           │                                                     │
   ┌───────▼────────┐                                     ┌──────▼─────────┐
   │     cli.py     │                                     │   ui_app.py    │
   │ Parser CLI      │                                     │ Interfaz Tkinter│
   │ - argparse       │                                     │ - Configura App │
   │ - arma AppConfig │                                     │ - Ejecuta hilo  │
   │ - lanza VideoProc│                                     │   VideoProcessor │
   └────────┬────────┘                                     └────────┬────────┘
            │                                                       │
            └───────────────────┬───────────────────────────────────┘
                                │
                        ┌───────▼────────────────────┐
                        │     processor.py            │
                        │ Orquestador principal       │
                        │ - Captura video (cv2)       │
                        │ - Detector (YOLO)           │
                        │ - Tracker (ByteTrack)       │
                        │ - Counter IN/OUT            │
                        │ - CSV y MLflow logging      │
                        │ - Render HUD + alertas beep │
                        └───────┬───────────────┬─────┘
                                │               │
      ┌─────────────────────────┘               └───────────────────────────┐
      │                                                                    │
┌─────▼─────┐   ┌─────────────────────┐   ┌────────────────────┐   ┌───────▼─────┐
│ config.py │   │    detector.py      │   │    counter.py       │   │ utils.py    │
│ - AppConfig│   │ VehicleDetector     │   │ LineCrossingCounter │   │ winsound_beep│
│ - parámetros│  │ - Carga YOLO        │   │ - IN/OUT por clase  │   │ (beep alerta)│
│ - csv, caps │  │ - Filtra car/moto   │   │ - Inventario        │   └──────────────┘
└────────────┘   └─────────────────────┘   └────────────────────┘
                                │
                                │
                        ┌───────▼─────────────────────┐
                        │ mlflow_integration.py        │
                        │ Tracker especializado        │
                        │ - Inicia/termina runs        │
                        │ - Registra params/metrics    │
                        │ - Guarda artefactos (CSV,    │
                        │   clases YOLO, gráficos)     │
                        └──────────────────────────────┘
```


### Servidor de inferencia
Levantar el servidor:
```bash
python services/inference_server.py
```
Escucha en `localhost:50051`.

### Cliente CLI
```bash
python clients/grpc_client.py "videos/prueba1.MP4"
```
Muestra progreso y devuelve el CSV.

### Cliente Streamlit
Próxima etapa: el `streamlit_app.py` se conecta al servidor gRPC y muestra los frames enviados.

---

## 📊 MLflow - Seguimiento de Experimentos

### 🚀 **Características MLflow Implementadas**

**✅ Integración Completa**
- **Habilitado por defecto** en todas las interfaces (Tkinter, Streamlit, CLI)
- **Seguimiento automático** de parámetros, métricas y artefactos
- **Interfaz web** para visualización y análisis
- **🆕 Model Registry** con versionado automático (Staging/Production)

**📈 Métricas Registradas**
- **Detección**: detecciones por frame, FPS, objetos por clase
- **Conteo**: entradas/salidas por tipo, inventario actual, flujo neto
- **Rendimiento**: tiempo de procesamiento, eficiencia, memoria
- **🆕 Validación**: precisión, recall, F1-score por clase

**🏷️ Parámetros Registrados**
- **Modelo**: arquitectura YOLO, umbrales de confianza
- **Sistema**: orientación de línea, capacidades, configuración
- **Video**: fuente, resolución, duración
- **🆕 Hardware**: CPU, GPU, RAM, CUDA, versiones de librerías

**📊 Artefactos Visuales** ✅ **IMPLEMENTADO v2.1**
- **Gráficos automáticos**: evolución de FPS, distribución de métricas
- **Dashboards de rendimiento**: 4 gráficos por experimento
- **Visualizaciones PNG**: guardadas automáticamente como artefactos
- **Reportes CSV** registrados automáticamente

**🔧 Mejoras de Interfaz** ✅ **IMPLEMENTADO v2.1**
- **Ventana única OpenCV**: eliminadas ventanas grises duplicadas
- **Gestión determinista**: nombres de ventana con timestamp
- **Limpieza robusta**: sin residuos al cerrar aplicación
- **Optimizado para Windows**: múltiples waitKey() calls

### 🌐 **Interfaz Web MLflow**

**🆕 Recomendado para Windows (evita bloqueo de consola):**
```powershell
# Opción 1: Script batch (más simple)
mlflow_ui.bat

# Opción 2: Script Python mejorado
uv run -p .venv python mlflow_ui_simple.py
```

**Opciones tradicionales:**
```powershell
# Script original (puede bloquear consola)
uv run -p .venv python launch_mlflow_ui.py

# Comando directo
mlflow ui --port 5000
```

**Accede a**: **http://localhost:5000**

💡 **Nota**: Si la consola se bloquea, usa `Ctrl+C` y luego `exit`

### 📋 **Uso**
MLflow funciona automáticamente:
- **Tkinter**: `python src/app.py` 
- **Streamlit**: `streamlit run streamlit_app.py`
- **CLI**: `python src/app.py --cli --webcam`

Todos los experimentos se registran automáticamente con métricas en tiempo real.

### 📖 **Documentación Completa**
Ver **[MLFLOW_INTEGRATION.md](MLFLOW_INTEGRATION.md)** para:
- Configuración avanzada
- Personalización de experimentos
- Análisis de métricas
- Cases de uso detallados

---

## 📌 Próximos pasos
- ~~Integración con **MLflow** para registrar parámetros, métricas y artefactos (CSV).~~ ✅ **IMPLEMENTADO**
- ~~**MLflow Model Registry** con versionado automático~~ ✅ **IMPLEMENTADO v2.1**
- ~~**Visualizaciones automáticas** de métricas~~ ✅ **IMPLEMENTADO v2.1**  
- ~~**Información del sistema** (hardware/software)~~ ✅ **IMPLEMENTADO v2.1**
- ~~**Ventana única OpenCV** sin duplicados~~ ✅ **IMPLEMENTADO v2.1**
- Configuración de **docker-compose** para levantar `inference` (gRPC) + `ui` (Streamlit).  
- Añadir **CI/CD** con GitHub Actions (pytest + coverage).  
- Documentar **flujo Gitflow y Kanban** en el repo.

---


---

## 📂 Estructura del Proyecto

```
Contador-de-Vehiculos/
├── src/                    # Código fuente principal
│   ├── app.py              # Entrada principal (CLI / UI Tkinter)
│   ├── cli.py              # Interfaz de línea de comandos
│   ├── config.py           # Configuración central (AppConfig)
│   ├── counter.py          # Lógica de conteo IN/OUT
│   ├── detector.py         # Detector de vehículos (YOLO)
│   ├── processor.py        # Procesador de video principal
│   ├── ui_app.py           # Interfaz Tkinter
│   ├── utils.py            # Utilidades generales
│   └── ...
│
├── clients/                # Clientes gRPC
│   └── grpc_client.py
│
├── services/               # Servidores gRPC
│   └── inference_server.py
│
├── proto/                  # Definiciones Protobuf
│   └── vehicle.proto
│
├── tests/                  # Pruebas automáticas (pytest)
│   ├── test_app_csv.py
│   ├── test_counter.py
│   ├── test_detector_mapping.py
│   ├── test_headless_integration.py
│   └── ...
│
├── reports/                # CSV generados (ignorado en git)
├── uploads/                # Videos subidos por UI Streamlit
├── mlruns/                 # Experimentos MLflow (ignorado en git)
│
├── streamlit_app.py        # Interfaz web alternativa
├── launch_mlflow_ui.py     # Script para abrir MLflow UI
├── Dockerfile              # Imagen Docker del proyecto
├── Makefile                # Automatización de comandos
├── requirements.txt        # Dependencias de producción
├── requirements-dev.txt    # Dependencias de desarrollo
└── README.md               # Este archivo
```

---

## 🏗️ Arquitectura General

El sistema sigue una arquitectura **modular y desacoplada**:

- **Procesador de Video (VideoProcessor)**  
  Orquesta detección, tracking y conteo.  
- **Detector (YOLO)**  
  Se encarga de obtener bounding boxes y clases.  
- **Counter (LineCrossingCounterByClass)**  
  Calcula IN/OUT por clase y mantiene inventario.  
- **Interfaces**  
  - CLI (`cli.py`)  
  - Tkinter (`ui_app.py`)  
  - Streamlit (`streamlit_app.py`)  
- **Servicios gRPC**  
  - `inference_server.py`: procesamiento  
  - `grpc_client.py`: cliente de pruebas  
- **MLflow**  
  - Registro automático de parámetros, métricas y CSV.  

---

## ⚡ Comandos útiles con Makefile

Además de los ejemplos que ya tienes en el README:

```bash
# Ejecutar interfaz Tkinter
make run-ui

# Ejecutar servidor gRPC
make serve

# Cliente gRPC (procesa un video y muestra progreso)
make grpc-client SRC="videos/prueba1.MP4"

# Construir imagen Docker
make docker-build

# Ejecutar CLI dentro de Docker (headless)
make docker-run-cli SRC="videos/prueba1.MP4"

# Formatear código (black + isort + ruff)
make format

# Limpiar archivos temporales y CSV
make clean
```

📌 Ejecuta `make help` para ver todos los comandos disponibles.  

---
