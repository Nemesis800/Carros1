# Detección, Seguimiento y Conteo de Vehículos en Tiempo Real (YOLOv11 + Supervision)

Este proyecto detecta, sigue y cuenta vehículos (carros y motos) en tiempo real a partir de un video cargado manualmente o la webcam. Mantiene un inventario por tipo de vehículo con capacidades configurables y genera una alarma visual y auditiva cuando se excede la capacidad definida para cada tipo.  
Ahora incluye **soporte para YOLOv8, YOLOv11 y YOLOv12**, generación de **reportes CSV configurables**, **modo CLI headless** y **pruebas automáticas con pytest**.

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

---

## Características
- Detección de vehículos en tiempo real con modelos YOLO (`yolov8n.pt`, `yolo11n.pt`, `yolo12n.pt`)
- Seguimiento multi-objeto (ByteTrack) para asignar ID por vehículo
- Conteo por cruce de línea con dirección IN/OUT por tipo (carro, moto)
- Inventario por tipo y alarma al exceder capacidad (visual + beep en Windows)
- **Reporte CSV de eventos y resumen final**
  - Configurable desde la UI: activar/desactivar, carpeta destino y nombre del archivo
  - Compatible con Excel (separador `;`, codificación UTF-8 BOM)
- **Modo CLI headless** para procesar videos largos sin abrir ventanas (ideal para servidores o procesamiento offline)
- **Pruebas unitarias y de integración** con pytest
- **Coverage report** para ver qué porcentaje del código está probado
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
- Webcam opcional para pruebas en vivo
- GPU NVIDIA opcional para acelerar la inferencia (CUDA/cuDNN)

---

## Instalación
```bash
# Crear entorno virtual (ejemplo con uv)
uv venv .venv
.\.venv\Scripts\activate

# Instalar dependencias
uv pip install -r requirements.txt
```

---

## Ejecución con Interfaz (UI)

```powershell
# Con UV
uv run -p .venv python src/app.py

# Con pip tradicional
.\.venv\Scripts\activate.bat
python src/app.py
```

---

## **Novedades en la UI**

- **Reporte CSV configurable**
  - ✅ Checkbox: activar o desactivar guardado de CSV  
  - ✅ Campo para seleccionar la carpeta de destino  
  - ✅ Campo para escribir el nombre del archivo (ej: `turno_mañana` → se guarda como `turno_mañana.csv`)  
  - Si lo dejas vacío, se genera automáticamente con timestamp y nombre del video  

- **Botón “Copiar comando CLI (headless)”**
  - Copia al portapapeles un comando listo para correr en terminal y procesar sin abrir ventanas  

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

> En ambos casos se genera un CSV con todos los eventos IN/OUT y un SUMMARY final.

---

## 📄 Reporte CSV

El archivo contiene columnas:

```
timestamp;evento;clase;car_in;car_out;moto_in;moto_out;car_inv;moto_inv;modelo;conf;orientacion;pos_linea;invertido;fuente
```

Ejemplo de filas:
```
2025-09-17T12:34:56;IN;car;1;0;0;0;1;0;yolo12n.pt;0.30;vertical;0.50;False;ejemplo.mp4
2025-09-17T12:35:12;OUT;motorcycle;1;0;0;1;1;-1;yolo12n.pt;0.30;vertical;0.50;False;ejemplo.mp4
2025-09-17T12:40:00;SUMMARY;-;15;10;4;5;5;-1;yolo12n.pt;0.30;vertical;0.50;False;ejemplo.mp4
```

---

## 🔬 Pruebas Automáticas

### 📁 Estructura
```
Contador-de-Vehiculos/
├─ src/
│  ├─ app.py        # Punto de entrada (UI o CLI)
│  ├─ cli.py        # CLI/headless
│  ├─ config.py     # Configuración central
│  ├─ processor.py  # Núcleo: detección → tracking → conteo → CSV
│  ├─ ui_app.py     # Interfaz gráfica Tkinter
│  ├─ utils.py      # Utilidades (beep cross-platform)
│  ├─ counter.py    # Lógica de conteo por cruce de línea
│  └─ detector.py   # Envoltorio YOLO con fallbacks y normalización
│
├─ tests/           # Pruebas unitarias e integración
├─ reports/         # CSV generados (auto)
└─ Makefile / Dockerfile
```

### ▶️ Ejecutar pruebas

```powershell
# Instalar dependencias de desarrollo
pip install -r requirements-dev.txt

# Correr pruebas
pytest -q

# Resultado esperado:
# 6 passed in X.XXs
```

### 📊 Coverage report

```powershell
coverage run -m pytest
coverage report -m
coverage html  # genera htmlcov/index.html
```

Abre `htmlcov/index.html` en tu navegador para ver líneas cubiertas.

---

## Cambios recientes

### ✅ Versión actual (Septiembre 2025)
- **Nuevo soporte para YOLOv12**
- **Reporte CSV configurable**:
  - Activar/desactivar desde la UI
  - Selección de carpeta
  - Campo para escribir el nombre del archivo
- **Modo CLI headless** para procesar videos sin interfaz gráfica
- **Pruebas unitarias y de integración** añadidas con pytest
- **Coverage report** habilitado para medir calidad de pruebas
- **Botón “Copiar comando CLI (headless)”** en la UI

### 📌 Cambios previos
- Mejora de la UI con Tkinter (sliders para línea, spinners de capacidad, etc.)
- Alarma visual y sonora cuando se excede la capacidad de carros o motos
- Exportación de reportes CSV con conteo de entradas, salidas e inventario
- Inclusión de `requirements.txt`, `CASOS_USO.md`, `CONTRIBUTING.md`
- Documentación inicial y guía de instalación

