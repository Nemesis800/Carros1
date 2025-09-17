# Detección, Seguimiento y Conteo de Vehículos en Tiempo Real (YOLOv11 + Supervision)

Este proyecto detecta, sigue y cuenta vehículos (carros y motos) en tiempo real a partir de un video cargado manualmente o la webcam. Mantiene un inventario por tipo de vehículo con capacidades configurables y genera una alarma visual y auditiva cuando se excede la capacidad definida para cada tipo.  
Ahora incluye **soporte para YOLOv8, YOLOv11 y YOLOv12**, generación de **reportes CSV configurables**, **modo CLI headless**, **pruebas automáticas con pytest**, además de **Dockerfile y Makefile** para simplificar despliegue y ejecución.

---

## Tecnologías
- YOLO v8/v11/v12 (Ultralytics) – detección de objetos
- Supervision – seguimiento (ByteTrack) y utilidades de anotación
- OpenCV (cv2) – lectura de video y visualización
- NumPy – operaciones numéricas
- Matplotlib – incluida como dependencia (opcional); para visualizaciones alternativas
- Tkinter – interfaz gráfica
- Pytest – pruebas unitarias e integración
- Coverage – reporte de cobertura
- Docker – empaquetado y ejecución en contenedores
- Make – automatización de comandos

---

## 📂 Estructura del proyecto

```
Contador-de-Vehiculos/
├─ .gitignore
├─ .dockerignore
├─ Dockerfile
├─ Makefile
├─ README.md
├─ requirements.txt
├─ requirements-dev.txt
├─ CASOS_USO.md
├─ CONTRIBUTING.md
├─ reports/                  # Carpeta ignorada en git (para CSV generados)
├─ src/
│   ├─ app.py
│   ├─ counter.py
│   └─ detector.py
└─ tests/
    ├─ test_counter.py
    ├─ test_app_csv.py
    ├─ test_detector_mapping.py
    └─ test_headless_integration.py
```

---

## Características
- Detección de vehículos en tiempo real con modelos YOLO (`yolov8n.pt`, `yolo11n.pt`, `yolo12n.pt`)
- Seguimiento multi-objeto (ByteTrack) para asignar ID por vehículo
- Conteo por cruce de línea con dirección IN/OUT por tipo (carro, moto)
- Inventario por tipo y alarma al exceder capacidad (visual + beep en Windows)
- **Reporte CSV de eventos y resumen final**
  - Configurable desde la UI: activar/desactivar, carpeta destino y nombre del archivo
  - Compatible con Excel (separador `;`, codificación UTF-8 BOM)
- **Modo CLI headless** para procesar videos largos sin abrir ventanas
- **Pruebas unitarias y de integración** con pytest
- **Coverage report** para ver qué porcentaje del código está probado
- **Soporte Docker y Makefile** para simplificar la ejecución
- Interfaz gráfica para:
  - Seleccionar video o usar webcam
  - Elegir orientación y posición de la línea
  - Invertir dirección de conteo
  - Configurar capacidades
  - Guardar reportes CSV con nombre definido por el usuario

---

## 💻 Requisitos del Sistema
- Python 3.11+
- Pip o [UV](https://github.com/astral-sh/uv) para manejar dependencias
- Windows, Linux o macOS (probado principalmente en Windows 10/11 y Ubuntu 22.04)
- Docker Desktop (para ejecutar con contenedores)
- GNU Make (para usar el Makefile en Windows instalar con WSL o Chocolatey)
- Webcam opcional para pruebas en vivo
- GPU NVIDIA opcional para acelerar la inferencia (CUDA/cuDNN)

---

## Instalación local
```bash
# Crear entorno virtual (ejemplo con uv)
uv venv .venv
.\.venv\Scriptsctivate

# Instalar dependencias
uv pip install -r requirements.txt
```

---

## Ejecución con Interfaz (UI)

```powershell
# Con UV
uv run -p .venv python src/app.py

# Con pip tradicional
.\.venv\Scriptsctivate.bat
python src/app.py
```

---

## **Novedades en la UI**

- **Reporte CSV configurable**
  - ✅ Checkbox: activar o desactivar guardado de CSV  
  - ✅ Campo para seleccionar la carpeta de destino  
  - ✅ Campo para escribir el nombre del archivo (ej: `turno_mañana.csv`)  
  - Si lo dejas vacío, se genera automáticamente con timestamp y nombre del video  

- **Botón “Copiar comando CLI (headless)”**
  - Copia un comando listo para correr en terminal y procesar sin abrir ventanas  

---

## Ejecución en Modo CLI (headless)

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

## 📄 Reporte CSV

El archivo contiene columnas:

```
timestamp;evento;clase;car_in;car_out;moto_in;moto_out;car_inv;moto_inv;modelo;conf;orientacion;pos_linea;invertido;fuente
```

Ejemplo:
```
2025-09-17T12:34:56;IN;car;1;0;0;0;1;0;yolo12n.pt;0.30;vertical;0.50;False;ejemplo.mp4
2025-09-17T12:35:12;OUT;motorcycle;1;0;0;1;1;-1;yolo12n.pt;0.30;vertical;0.50;False;ejemplo.mp4
2025-09-17T12:40:00;SUMMARY;-;15;10;4;5;5;-1;yolo12n.pt;0.30;vertical;0.50;False;ejemplo.mp4
```

---

## 🔬 Pruebas Automáticas

### 📁 Estructura de pruebas
```
tests/
├─ test_counter.py          # Conteo y cruces
├─ test_app_csv.py          # CSV (eventos y summary)
├─ test_detector_mapping.py # Normalización de clases
└─ test_headless_integration.py # Integración headless
```

### ▶️ Ejecutar
```powershell
pip install -r requirements-dev.txt
pytest -q
# 6 passed in X.XXs
```

### 📊 Coverage
```powershell
coverage run -m pytest
coverage report -m
coverage html
```
Abre `htmlcov/index.html` en el navegador.

---

## 🚀 Uso con Docker

### Construir imagen
```powershell
make docker-build
```

### Ejecutar con video
```powershell
make docker-run-cli SRC="C:/ruta/video.mp4" MODEL=yolo12n.pt CSV_NAME=turno_noche CONF=0.30 ORIENT=vertical LINE_POS=0.50 CAP_CAR=80 CAP_MOTO=60
```

### Ejecutar con webcam
```powershell
make docker-run-cli WEBCAM=1 MODEL=yolo12n.pt CSV_NAME=webcam_test
```

> Los reportes se guardan en la carpeta `reports/` del host.

---

## 🛠️ Uso con Makefile (local)

### Crear entorno e instalar deps
```powershell
make venv
make install
make install-dev
```

### Ejecutar en local
```powershell
make run-ui
make run-cli SRC="C:/ruta/video.mp4" MODEL=yolo11n.pt CSV_NAME=prueba_local
```

### Ejecutar pruebas
```powershell
make test
make cov
```

---

## Cambios recientes

### ✅ Versión actual (Octubre 2025)
- **Nuevo soporte para Dockerfile** (ejecución en contenedor)
- **Nuevo Makefile** con targets para build, run, tests y coverage
- **Estructura del proyecto actualizada** (incluye Docker y reports/ ignorado en git)
- Mejor compatibilidad de Makefile con `cmd.exe` en Windows
- Correcciones para modo CLI headless sin dependencias de Tkinter

### 📌 Cambios previos
- Soporte para YOLOv12
- Reporte CSV configurable (activar/desactivar, carpeta, nombre archivo)
- Modo CLI headless
- Pruebas unitarias y de integración con pytest
- Coverage report
- Botón “Copiar comando CLI (headless)” en la UI
- Mejora de la UI con Tkinter (sliders para línea, spinners de capacidad, etc.)
- Alarma visual y sonora al exceder capacidad
- Documentación inicial y guía de instalación
