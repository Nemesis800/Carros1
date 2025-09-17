# Detección, Seguimiento y Conteo de Vehículos en Tiempo Real  
*(YOLOv8/v11/v12 + Supervision)*

Este proyecto detecta, sigue y cuenta vehículos (carros y motos) en tiempo real a partir de un video o webcam.  
Mantiene un inventario por tipo de vehículo, genera reportes CSV configurables y lanza una alarma visual/sonora al exceder la capacidad definida.  

Incluye:  
- **Soporte YOLOv8, YOLOv11 y YOLOv12**  
- **Reportes CSV configurables**  
- **Modo CLI headless**  
- **Pruebas automáticas (pytest + coverage)**  
- **Dockerfile y Makefile** para despliegue y ejecución simplificada  

---

## ⚙️ Tecnologías principales
- **YOLO v8/v11/v12** (Ultralytics) → detección de objetos  
- **Supervision** → seguimiento (ByteTrack) y anotaciones  
- **OpenCV (cv2)** → lectura/visualización de video  
- **Tkinter** → interfaz gráfica  
- **Pytest + Coverage** → pruebas y reportes  
- **Docker + Make** → empaquetado y automatización  

---

## 📂 Estructura del Proyecto
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

---

## 🧩 Funcionalidades clave
- 🚦 **Detección y seguimiento** de carros y motos en tiempo real  
- 🔢 **Conteo por línea de cruce** (IN/OUT por clase)  
- 📊 **Inventario dinámico** + alarma al exceder capacidad  
- 📝 **CSV por evento + fila SUMMARY**  
- 🖥️ **UI con Tkinter** para parametrización  
- 🖲️ **CLI headless** para procesar videos largos sin ventanas  
- 🧪 **Pruebas automáticas** y reporte de cobertura  

---

## 💻 Requisitos
- Python 3.11+  
- Pip o [UV](https://github.com/astral-sh/uv)  
- Windows / Linux / macOS  
- Docker Desktop + GNU Make (opcional)  
- Webcam y GPU NVIDIA (opcionales)  

---

## 🚀 Instalación
```bash
# Crear entorno virtual
uv venv .venv
.\.venv\Scripts\activate

# Instalar dependencias
uv pip install -r requirements.txt
```

---

## ▶️ Ejecución

### Interfaz gráfica
```bash
uv run -p .venv python src/app.py
# o con pip
.\.venv\Scripts\activate.bat
python src/app.py
```

### Modo CLI (headless)
Con webcam:
```bash
python src/app.py --cli --webcam --model yolo12n.pt --conf 0.30 ^
  --orientation vertical --line-pos 0.50 ^
  --cap-car 50 --cap-moto 50 ^
  --csv --csv-dir reports --csv-name "turno_noche" --no-display
```

Con archivo de video:
```bash
python src/app.py --cli --source "C:\Videos\ejemplo.mp4" --model yolo11n.pt ^
  --orientation horizontal --line-pos 0.25 --invert ^
  --csv --csv-dir reports --csv-name "parqueadero_sabado" --no-display
```

---

## 📄 Formato del Reporte CSV
Columnas:
```
timestamp;evento;clase;car_in;car_out;moto_in;moto_out;car_inv;moto_inv;
modelo;conf;orientacion;pos_linea;invertido;fuente
```

Ejemplo:
```
2025-09-17T12:34:56;IN;car;1;0;0;0;1;0;yolo12n.pt;0.30;vertical;0.50;False;ejemplo.mp4
2025-09-17T12:40:00;SUMMARY;-;15;10;4;5;5;-1;yolo12n.pt;0.30;vertical;0.50;False;ejemplo.mp4
```

---

## 🧪 Pruebas
```bash
pip install -r requirements-dev.txt
pytest -q
coverage run -m pytest
coverage report -m
coverage html  # abrir htmlcov/index.html
```

---

## 🐳 Uso con Docker
```bash
# Construir imagen
make docker-build

# Ejecutar con video
make docker-run-cli SRC="C:/ruta/video.mp4" MODEL=yolo12n.pt CSV_NAME=turno_noche

# Ejecutar con webcam
make docker-run-cli WEBCAM=1 MODEL=yolo12n.pt CSV_NAME=webcam_test
```

---

## 🛠️ Uso con Makefile
```bash
make venv
make install
make install-dev
make run-ui
make run-cli SRC="C:/ruta/video.mp4" MODEL=yolo11n.pt CSV_NAME=prueba_local
make test
make cov
```

---

## 📌 Cambios recientes
### ✅ Octubre 2025
- Soporte Dockerfile  
- Makefile con targets para build, run y pruebas  
- Mejor compatibilidad en Windows (`cmd.exe`)  
- Correcciones CLI headless sin Tkinter  

### 📌 Anteriores
- Soporte YOLOv12  
- Reportes CSV configurables  
- UI mejorada (sliders, spinners, botón *Copiar comando CLI*)  
- Alarmas visuales/sonoras  
- Pruebas + Coverage report  
