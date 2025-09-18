# Integración MLflow - Sistema de Detección de Vehículos

## 🎯 Descripción General

Este proyecto integra **MLflow** para el seguimiento completo de experimentos, registro de modelos y gestión de artefactos en el sistema de detección y conteo de vehículos.

## 🚀 Características de la Integración

### 📊 **Seguimiento de Experimentos**
- **Parámetros de configuración**: modelo YOLO, umbrales, orientación de línea, capacidades
- **Métricas en tiempo real**: detecciones por frame, FPS, conteos de vehículos
- **Métricas de rendimiento**: tiempo de procesamiento, eficiencia, memoria
- **Tags automáticos**: tipo de video, modelo, timestamp

### 🤖 **Registro de Modelos**
- **Metadatos del modelo**: arquitectura, clases, tamaño de entrada
- **Artefactos**: archivos .pt del modelo YOLO
- **Información de clases**: clases objetivo (car, motorcycle)
- **🆕 Model Registry**: versionado automático con estados Staging/Production
- **🆕 Registro automático**: modelos YOLO registrados al finalizar cada experimento

### 📁 **Gestión de Artefactos**
- **Reportes CSV**: archivos de eventos y resúmenes
- **Muestras de video**: videos pequeños como referencia
- **Logs de configuración**: parámetros y metadatos
- **🆕 Gráficos de rendimiento**: visualizaciones automáticas de FPS y métricas
- **🆕 Información del sistema**: hardware, versiones de software, configuración
- **🆕 Dashboards visuales**: evolución de métricas y resumen de experimento

## 🔧 Configuración y Uso

### 1. **Instalación**
MLflow se instala automáticamente con las dependencias:
```powershell
uv pip install -r requirements.txt
```

### 2. **Uso Automático**
La integración MLflow está **habilitada por defecto** en todas las interfaces:

**Tkinter:**
```powershell
cd src
uv run -p ../.venv python app.py
```

**Streamlit:**
```powershell
uv run -p .venv streamlit run streamlit_app.py
```

**CLI:**
```powershell
cd src
uv run -p ../.venv python app.py --cli --webcam
```

### 3. **Parámetros de MLflow**
Puedes personalizar la integración MLflow:

```python
# En el código Python
processor = VideoProcessor(
    video_source=0,
    config=config,
    stop_event=stop_event,
    enable_mlflow=True,                    # Habilitar/deshabilitar MLflow
    experiment_name="mi_experimento",      # Nombre del experimento
    mlflow_tags={"usuario": "admin"}       # Tags personalizados
)
```

### 4. **Interfaz Web de MLflow**
Para ver los experimentos y métricas:

```powershell
# Opción 1: Usar script incluido
uv run -p .venv python launch_mlflow_ui.py

# Opción 2: Comando directo
mlflow ui --port 5000
```

Accede a: **http://localhost:5000**

## 📈 Métricas Registradas

### **Métricas de Detección** (cada 30 frames)
- `detections_per_frame`: Número de detecciones por frame
- `cars_detected`: Carros detectados en el frame
- `motorcycles_detected`: Motos detectadas en el frame
- `current_fps`: FPS actual del procesamiento
- `avg_fps`: FPS promedio acumulado
- `total_detections`: Total de detecciones acumuladas
- `avg_detections_per_frame`: Promedio de detecciones por frame

### **Métricas de Conteo** (cada 60 frames)
- `cars_in_total`: Total de carros que entraron
- `cars_out_total`: Total de carros que salieron
- `cars_inventory`: Inventario actual de carros
- `motorcycles_in_total`: Total de motos que entraron
- `motorcycles_out_total`: Total de motos que salieron
- `motorcycles_inventory`: Inventario actual de motos
- `total_vehicles_inside`: Total de vehículos dentro
- `total_entries`: Total de entradas
- `total_exits`: Total de salidas
- `net_flow`: Flujo neto de vehículos
- `capacity_exceeded`: Indicador de capacidad excedida

### **Métricas Finales**
- `total_processing_time`: Tiempo total de procesamiento
- `total_frames_processed`: Total de frames procesados
- `total_detections_final`: Total final de detecciones
- `final_avg_fps`: FPS promedio final
- `processing_efficiency`: Eficiencia del procesamiento

## 🏷️ Parámetros Registrados

### **Configuración del Modelo**
- `model_name`: Nombre del modelo YOLO (yolo11n.pt, yolo12n.pt, etc.)
- `confidence_threshold`: Umbral de confianza
- `iou_threshold`: Umbral IoU
- `model_type`: Tipo de modelo (YOLO)
- `model_architecture`: Arquitectura específica
- `input_size`: Tamaño de entrada (640)
- `num_classes`: Número de clases del modelo

### **Configuración del Sistema**
- `line_orientation`: Orientación de línea (horizontal/vertical)
- `line_position`: Posición de línea (0.1-0.9)
- `invert_direction`: Dirección invertida (true/false)
- `capacity_car`: Capacidad máxima de carros
- `capacity_moto`: Capacidad máxima de motos
- `device`: Dispositivo de procesamiento (CPU/GPU)
- `video_source`: Fuente de video
- `display_mode`: Modo de visualización

## 📊 Visualización en MLflow UI

### **Página de Experimentos**
- Lista de todos los experimentos ejecutados
- Comparación de métricas entre runs
- Filtros por tags, parámetros y métricas

### **Página de Run Individual**
- **Overview**: Resumen del experimento
- **Metrics**: Gráficos de métricas en tiempo real
- **Parameters**: Todos los parámetros de configuración
- **Tags**: Metadatos del experimento
- **Artifacts**: Archivos CSV, modelos, logs

### **Gráficos Útiles**
- **FPS vs Tiempo**: Rendimiento del sistema
- **Detecciones por Frame**: Actividad de detección
- **Inventario de Vehículos**: Ocupación en tiempo real
- **Flujo Neto**: Entrada vs salida de vehículos

## 🔍 Casos de Uso

### **Optimización de Parámetros**
Compara diferentes configuraciones para encontrar:
- Mejor umbral de confianza para tu escenario
- Posición óptima de línea de conteo
- Modelo YOLO más eficiente

### **Análisis de Rendimiento**
Monitorea:
- FPS bajo diferentes condiciones
- Uso de memoria y recursos
- Eficiencia de procesamiento

### **Seguimiento de Experimentos**
Registra:
- Diferentes videos de prueba
- Configuraciones de línea
- Comparaciones de modelos

### **Reportes y Auditoría**
Genera:
- Informes de precisión de conteo
- Estadísticas de uso del sistema
- Trazabilidad de configuraciones

## 🛠️ Personalización Avanzada

### **Deshabilitar MLflow**
```python
processor = VideoProcessor(
    video_source=0,
    config=config,
    stop_event=stop_event,
    enable_mlflow=False  # Deshabilitar MLflow
)
```

### **Experimentos Personalizados**
```python
processor = VideoProcessor(
    video_source="video.mp4",
    config=config,
    stop_event=stop_event,
    experiment_name="experimento_parking_mall",
    mlflow_tags={
        "location": "mall",
        "camera": "entrance",
        "weather": "sunny",
        "version": "v2.1"
    }
)
```

### **Servidor MLflow Remoto**
Para usar un servidor MLflow remoto, modifica el código en `processor.py`:
```python
# En _init_mlflow():
mlflow.set_tracking_uri("http://mlflow-server:5000")
```

## 📋 Troubleshooting

### **MLflow no disponible**
Si MLflow no está instalado, el sistema funcionará normalmente sin logging:
```
⚠️  MLflow no está instalado. Las métricas no se registrarán.
```

### **Error de permisos en directorio mlruns**
Asegúrate de que el directorio tenga permisos de escritura:
```powershell
mkdir mlruns
```

### **Puerto ocupado para MLflow UI**
Cambia el puerto en el comando:
```powershell
python launch_mlflow_ui.py --port 5001
```

### **Muchas métricas registradas**
Las métricas se registran cada 30-60 frames para evitar saturación. Para cambiar la frecuencia, modifica en `processor.py`:
```python
# Cambiar frecuencia de logging
if self.frame_count % 10 == 0:  # Cada 10 frames
```

## 🆕 **Nuevas Características Implementadas (v2.1)**

### **1. MLflow Model Registry** ✅
- **Versionado automático** de modelos YOLO
- **Estados de modelo**: Staging → Production
- **Registro automático** al finalizar cada experimento

```python
# Automático - no requiere configuración adicional
# Los modelos se registran como "VehicleDetectionModel" v1, v2, etc.
```

### **2. Métricas de Validación** ✅
- **Precisión, Recall, F1-Score** por clase (car, motorcycle)
- **Métricas comparativas** entre diferentes runs
- **Evaluación objetiva** del rendimiento del modelo

### **3. Información Detallada del Sistema** ✅
- **Hardware**: CPU, RAM, GPU, CUDA
- **Software**: versiones de PyTorch, OpenCV, Ultralytics
- **Sistema**: OS, arquitectura, hostname
- **Registro automático** en cada experimento

### **4. Visualizaciones Automáticas** ✅
- **Gráficos de FPS**: evolución y distribución
- **Dashboard de rendimiento**: tiempo, frames, detecciones
- **Resúmenes visuales**: automáticamente guardados como PNG
- **Registro como artefactos** en MLflow

### **Cómo Acceder a las Nuevas Características**

1. **Model Registry**:
   ```
   MLflow UI → Models → VehicleDetectionModel → Versiones
   ```

2. **Visualizaciones**:
   ```
   MLflow UI → Experiments → [Tu Run] → Artifacts → visualizations/
   ```

3. **Info del Sistema**:
   ```
   MLflow UI → Experiments → [Tu Run] → Parameters → system_*
   ```

---

## 🚀 Próximas Mejoras (Roadmap)

- **Integración con bases de datos**: PostgreSQL, MySQL
- **Alertas automáticas**: Slack, email cuando se exceda capacidad  
- **Comparación automática**: A/B testing entre modelos
- **Métricas de IoU**: Comparación precisa con ground truth
- **Dashboard personalizado**: Visualizaciones específicas del dominio
- **Detección de anomalías**: Alertas por comportamiento inusual
