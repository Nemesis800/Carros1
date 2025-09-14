# Detección, Seguimiento y Conteo de Vehículos en Tiempo Real (YOLOv11 + Supervision)

Este proyecto detecta, sigue y cuenta vehículos (carros y motos) en tiempo real a partir de un video cargado manualmente o la webcam. Mantiene un inventario por tipo de vehículo con capacidades configurables y genera una alarma visual y auditiva cuando se excede la capacidad definida para cada tipo.

Tecnologías:
- YOLO v11 (Ultralytics) – detección de objetos con modelo ligero (yolo11n.pt)
- Supervision – seguimiento (ByteTrack) y utilidades de anotación
- OpenCV (cv2) – lectura de video y visualización
- NumPy – operaciones numéricas
- Matplotlib – incluida como dependencia (opcional); para visualizaciones alternativas
- Tkinter – interfaz gráfica para seleccionar el video y configurar la ejecución

## Características
- Detección de vehículos en tiempo real con el modelo liviano `yolo11n.pt` (se descarga automáticamente la primera vez).
- Seguimiento multi-objeto (ByteTrack) para asignar ID por vehículo.
- Conteo por cruce de línea con dirección IN/OUT, independiente por tipo (carro, moto).
- Inventario por tipo y alarma al exceder capacidad (visual + beep en Windows).
- Interfaz gráfica para:
  - Seleccionar video o usar webcam
  - Elegir orientación de la línea (horizontal/vertical)
  - **Ajustar la posición de la línea de conteo (10% a 90%)**
  - Invertir la dirección de conteo (IN ↔ OUT)
  - Configurar capacidades para carros y motos
- **Mensaje "Presiona Q para salir" visible en el video**

## 💻 Requisitos del Sistema

### **Software requerido:**
- **Python**: 3.10, 3.11 o 3.12 (recomendado 3.11+)
- **Sistema operativo**: 
  - ✅ Windows 10/11 (probado y optimizado)
  - ✅ Linux (Ubuntu, Debian, etc.)
  - ✅ macOS (el beep de alerta puede variar)
- **Gestor de paquetes**: UV (recomendado) o pip tradicional

### **Hardware recomendado:**
- **RAM**: Mínimo 4GB, recomendado 8GB+
- **Procesador**: Cualquier CPU moderna (i3/AMD equivalente+)
- **GPU**: Opcional pero recomendada para mejor rendimiento (NVIDIA con CUDA)
- **Webcam**: Opcional, para detección en tiempo real

## Instalación

### Opción 1: Usando UV (recomendado)
```powershell
# Instalar UV si no lo tienes
pip install uv

# Crear entorno virtual e instalar dependencias
uv venv .venv
uv pip install -r requirements.txt
```

### Opción 2: Usando pip tradicional
1. Crear y activar un entorno virtual:
   - PowerShell (Windows):
     ```powershell
     py -3.12 -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
2. Actualiza pip e instala dependencias:
   ```powershell
   python -m pip install -U pip
   pip install -r requirements.txt
   ```

Notas GPU: Ultralytics instalará PyTorch CPU por defecto. Si tienes GPU NVIDIA y quieres acelerar, instala PyTorch CUDA según tu sistema desde https://pytorch.org/get-started/locally/ y luego reinstala `ultralytics` si es necesario.

## Ejecución

### Con UV:
```powershell
uv run -p .venv python src/app.py
```

### Con pip tradicional:
```powershell
# Asegúrate de tener el entorno activado
.\.venv\Scripts\Activate.ps1
python src/app.py
```

## Configuraciones de la Aplicación

### Fuente de Video
- **Usar webcam (ID 0)**: Activa la cámara web predeterminada de tu computadora
- **Seleccionar video**: Permite cargar archivos de video en formatos MP4, AVI, MOV, MKV

### Configuración de Detección

#### Modelo
- **yolo11n.pt**: Modelo nano (más rápido, menos preciso) - Recomendado para tiempo real
- **yolov8n.pt**: Modelo YOLOv8 nano (alternativa si prefieres la versión anterior)
- El modelo se descarga automáticamente la primera vez que se usa

#### Conf (Confianza)
- **Rango**: 0.0 a 1.0 (controlado por slider)
- **Valor por defecto**: 0.3
- **Función**: Umbral mínimo de confianza para aceptar una detección
- **Recomendaciones**:
  - Valores bajos (0.2-0.3): Detecta más objetos pero puede tener falsos positivos
  - Valores medios (0.3-0.5): Balance entre detecciones y precisión
  - Valores altos (0.5-0.7): Solo detecta objetos con alta certeza

#### Orientación línea
- **horizontal**: La línea de conteo cruza horizontalmente el video (útil para entradas/salidas laterales)
- **vertical**: La línea de conteo cruza verticalmente el video (útil para entradas/salidas superiores/inferiores)
- **Valor por defecto**: vertical

#### Posición línea (Slider)
- **Rango**: 10% a 90% de la pantalla
- **Valor por defecto**: 50% (centro)
- **Función**: Permite mover la línea de conteo a lo largo del video
- **Comportamiento**:
  - Con línea **vertical**: El porcentaje controla la posición horizontal (10%=izquierda, 90%=derecha)
  - Con línea **horizontal**: El porcentaje controla la posición vertical (10%=arriba, 90%=abajo)
- **Casos de uso**:
  - Colocar la línea al 20-30% para detectar vehículos al entrar
  - Colocar la línea al 70-80% para detectar vehículos al salir
  - Ajustar para enfocarse en un carril específico
  - Evitar obstáculos o zonas problemáticas del video

#### Invertir dirección (IN<->OUT)
- **Sin marcar**: El cruce de izquierda a derecha (o de arriba a abajo) cuenta como entrada (IN)
- **Marcado**: Invierte la lógica de conteo (útil si la cámara está en sentido opuesto)

#### Capacidad carros
- **Rango**: 1 a 999
- **Valor por defecto**: 50
- **Función**: Número máximo de carros permitidos en el inventario
- **Alerta**: Se activa alarma visual y sonora cuando se excede

#### Capacidad motos
- **Rango**: 1 a 999  
- **Valor por defecto**: 50
- **Función**: Número máximo de motos permitidas en el inventario
- **Alerta**: Se activa alarma visual y sonora cuando se excede

### Uso de la Interfaz

#### **🚀 Flujo de trabajo recomendado:**
1. **Configuración inicial:**
   - Selecciona la fuente de video (archivo o webcam)
   - Ajusta orientación de línea (vertical/horizontal)
   - Configura posición de línea (10%-90%) según tu caso de uso
   - Establece capacidades para carros y motos
   - Ajusta confianza si es necesario (0.3 por defecto es bueno)

2. **Ejecución:**
   - Pulsa "**Iniciar**" para comenzar el procesamiento
   - Se abrirá una ventana con:
     - Video en tiempo real con detecciones
     - Cajas delimitadoras con ID de seguimiento y confianza
     - Línea de conteo (amarilla) en la posición configurada
     - Panel de estado con contadores IN/OUT e inventario por tipo
     - **Mensaje "Presiona Q para salir" en la esquina superior derecha**

3. **Control durante la ejecución:**
   - **Presiona `Q`** en la ventana del video = **DETIENE COMPLETAMENTE**
   - O usa el botón "**Detener**" en la interfaz principal
   - Ambas opciones tienen el mismo efecto

4. **Después de detener:**
   - ✅ Puedes cambiar el video de origen inmediatamente
   - ✅ Modificar cualquier configuración
   - ✅ Presionar "Iniciar" nuevamente sin problemas
   - ✅ No necesitas hacer nada especial entre ejecuciones

#### **💡 Tips de uso:**
- **Para probar videos diferentes:** Presiona 'Q' → Selecciona nuevo video → "Iniciar"
- **Para ajustar configuración:** 'Q' → Modifica parámetros → "Iniciar"
- **Para cambiar posición de línea:** 'Q' → Mueve slider → "Iniciar"

## ¿Cómo funciona el conteo?
- Se traza una línea en la posición configurada (10% a 90% del video):
  - **Horizontal**: Cruza de lado a lado en la altura seleccionada
  - **Vertical**: Cruza de arriba a abajo en la posición horizontal seleccionada
- Para cada objeto rastreado (con un ID), se calcula el centro del bounding box y se evalúa el lado de la línea (signo del producto cruzado). Cuando el signo cambia, se detecta un cruce:
  - Cruce de lado negativo a positivo = IN (por defecto)
  - Cruce de lado positivo a negativo = OUT (por defecto)
  - Puedes invertir esta lógica con “Invertir dirección”.
- Se mantienen contadores IN/OUT e inventarios por tipo (car/motorcycle). La alarma se activa cuando el inventario supera la capacidad configurada.

## Personalización
- **Modelo**: por defecto `yolo11n.pt` (ligero). Puedes cambiar a modelos más grandes (p. ej., `yolo11s.pt`) si están disponibles, actualizando el valor en la interfaz o en el código.
- **Confianza/IoU**: ajustables (confianza desde la UI; IoU está fijada a 0.5 en el código por simplicidad).
- **Línea de conteo**: Totalmente ajustable mediante el slider de posición (10% a 90%). La orientación (horizontal/vertical) también es configurable.
- **Posición de línea**: Ajustable en tiempo real antes de iniciar el procesamiento, permitiendo adaptarse a diferentes escenarios y ángulos de cámara.

## Limitaciones y recomendaciones
- La precisión depende del ángulo de cámara, iluminación y oclusiones. Si la cámara está muy inclinada o hay múltiples carriles cruzando, considera agregar varias líneas o zonas.
- Matplotlib está incluida pero no se usa en tiempo real (OpenCV es más eficiente). Puedes usarla para análisis offline de fotogramas.
- Para alto rendimiento:
  - Usa GPU (CUDA) si está disponible.
  - Mantén el modelo ligero (yolo11n) y/o reduce la resolución del video.

## Estructura del proyecto
```
Aplicacion/
├─ requirements.txt
├─ README.md
└─ src/
   ├─ app.py            # Interfaz + loop de video
   ├─ detector.py       # Capa de detección YOLOv11 (Ultralytics)
   └─ counter.py        # Conteo por cruce de línea por clase
```

## Seguridad y privacidad
- Todo el procesamiento se realiza en tu equipo. No se envían videos ni datos a la nube.

## ⚠️ Problemas resueltos

### ✅ **Video no se detiene correctamente (SOLUCIONADO)**
- **Problema anterior**: Al presionar 'Q' el video se cerraba pero la aplicación seguía ejecutándose
- **Problema anterior**: Al detener y reiniciar aparecía una "ventana gris" no funcional
- **✅ Solución actual**: Ambos problemas están completamente resueltos en la versión actual

### ✅ **Sincronización de threads (SOLUCIONADO)**
- **Problema anterior**: Los botones no reflejaban el estado real del procesamiento
- **✅ Solución actual**: Los botones y estados se actualizan correctamente en tiempo real

## Problemas comunes y soluciones

### Errores de sintaxis al ejecutar
- **SyntaxError con try/except**: Verificar la indentación del código
- **Solución**: Los bloques try deben tener la indentación correcta

### Errores de compatibilidad con supervision
- **"ColorPalette has no attribute 'default'"**: Versión incompatible de supervision
- **Solución**: El código ya está adaptado para supervision 0.26.1

### Problemas de video
- **No abre el video**: Verifica la ruta o el códec del archivo
- **Solución**: Usar formatos estándar (MP4 con H.264)

### Rendimiento
- **Baja velocidad de procesamiento**:
  - Instala PyTorch con CUDA si tienes GPU NVIDIA
  - Usa el modelo más ligero (yolo11n.pt)
  - Reduce la resolución del video si es muy alta

### Sistema operativo
- **No suena el beep**: En Linux/macOS el beep puede no funcionar
- **Solución**: La alerta visual siempre está disponible en pantalla

## Cambios recientes

### ✅ **Versión actual (Enero 2025)**
- **🔧 Mejora crítica del control de video**: 
  - Al presionar **'Q'** ahora se detiene completamente la aplicación (igual que el botón "Detener")
  - Eliminado el problema de "ventana gris" al reiniciar
  - Sincronización mejorada entre threads para evitar bloqueos
  - Limpieza automática de recursos OpenCV
- **🎮 Control de flujo mejorado**:
  - Después de presionar 'Q' puedes cambiar video y configuración inmediatamente
  - No necesitas presionar "Detener" después de usar 'Q'
  - Los botones reflejan correctamente el estado del procesamiento

### **Versiones anteriores**
- **Nuevo slider de posición de línea**: Permite ajustar la línea de conteo entre 10% y 90% del video
- **Mensaje "Presiona Q para salir"**: Ahora visible en la esquina superior derecha del video
- **Orientación vertical por defecto**: La línea ahora aparece vertical por defecto
- Corrección de errores de indentación en el bloque try/except
- Adaptación para compatibilidad con supervision 0.26.1
- Simplificación del sistema de colores para mayor compatibilidad
- Soporte para Python 3.12 y UV como gestor de paquetes
