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
El proyecto incluye un `Makefile` para simplificar tareas comunes:
```bash
# Ejecutar en modo CLI con video de ejemplo
make run-cli SRC="videos/prueba1.MP4" MODEL=yolo12n.pt CONF=0.3 ORIENT=vertical LINE_POS=0.5

# Ejecutar en Docker en modo CLI
make docker-run-cli SRC="videos/prueba1.MP4"

# Ejecutar pruebas
make test

# Generar reporte de coverage
make coverage
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

## 🧪 Pruebas
```bash
# Instalar dependencias de desarrollo
pip install -r requirements-dev.txt

# Correr pruebas
pytest -q

# Reporte de coverage
coverage run -m pytest
coverage report -m
coverage html
```
Abre `htmlcov/index.html` en tu navegador para ver cobertura.

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
```powershell
# Lanzar interfaz MLflow (script incluido)
uv run -p .venv python launch_mlflow_ui.py

# O directamente
mlflow ui --port 5000
```
Accede a: **http://localhost:5000**

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
