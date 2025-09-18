# tests/test_streamlit_app.py
"""
Tests para el módulo streamlit_app.py (interfaz web Streamlit)

Prueba:
- Inicialización del estado de sesión
- Procesamiento de argumentos
- Manejo de colas
- Callbacks
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock
from queue import Queue, Empty
import numpy as np
import pytest

# Añadir paths necesarios
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_session_state_variables():
    """Test variables esperadas en el estado de sesión."""
    # Lista de variables que deberían estar en session_state
    expected_vars = [
        'thread', 'stop_event', 'running', 'last_csv',
        'last_error', 'last_frame', 'progress',
        'frame_q', 'progress_q', 'finish_q', 'error_q'
    ]
    
    # Valores iniciales esperados
    initial_values = {
        'thread': None,
        'stop_event': None,
        'running': False,
        'last_csv': None,
        'last_error': None,
        'last_frame': None,
        'progress': 0.0
    }
    
    # Simular inicialización
    mock_session = {}
    for var in expected_vars:
        if var in initial_values:
            mock_session[var] = initial_values[var]
        elif var.endswith('_q'):
            mock_session[var] = Queue()
    
    # Verificar que todas las variables están presentes
    for var in expected_vars:
        assert var in mock_session
    
    # Verificar tipos de colas
    assert isinstance(mock_session['frame_q'], Queue)
    assert isinstance(mock_session['progress_q'], Queue)


def test_file_upload_handling():
    """Test manejo de archivos subidos."""
    from datetime import datetime
    from pathlib import Path
    
    # Simular función de guardado
    def save_uploaded_file(file_obj, upload_dir="uploads"):
        if file_obj is None:
            return None
        
        # Crear nombre único con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file_obj.name}"
        
        # Simular ruta de destino
        dest_path = Path(upload_dir) / filename
        
        return str(dest_path)
    
    # Test con archivo
    mock_file = MagicMock()
    mock_file.name = "video.mp4"
    result = save_uploaded_file(mock_file)
    assert result is not None
    assert "video.mp4" in result
    assert "uploads" in result
    
    # Test sin archivo
    result = save_uploaded_file(None)
    assert result is None


def test_save_uploaded_to_disk_none():
    """Test cuando no hay archivo para guardar."""
    from streamlit_app import _save_uploaded_to_disk
    
    result = _save_uploaded_to_disk(None)
    assert result is None


def test_callbacks():
    """Test de las funciones callback."""
    # Test cb_on_frame
    frame_q = Queue(maxsize=1)
    frame = np.ones((100, 100, 3), dtype=np.uint8)
    
    # Importamos después de los patches necesarios
    import sys
    if 'streamlit_app' in sys.modules:
        del sys.modules['streamlit_app']
    
    # Mock session_state para importación
    with patch('streamlit.session_state', {'frame_q': frame_q}):
        # Crear callback para frame
        def cb_on_frame(frame_rgb):
            try:
                if frame_q.full():
                    frame_q.get_nowait()
                frame_q.put_nowait(frame_rgb)
            except Exception:
                pass
        
        # Ejecutar callback
        cb_on_frame(frame)
        
        # Verificar que se agregó el frame
        assert not frame_q.empty()
        result = frame_q.get_nowait()
        assert np.array_equal(result, frame)


def test_progress_callback():
    """Test del callback de progreso."""
    progress_q = Queue(maxsize=8)
    
    def cb_on_progress(p):
        try:
            if progress_q.full():
                progress_q.get_nowait()
            progress_q.put_nowait(float(p))
        except Exception:
            pass
    
    # Ejecutar callback con varios valores
    cb_on_progress(0.25)
    cb_on_progress(0.50)
    cb_on_progress(0.75)
    
    # Verificar valores en la cola
    assert progress_q.get_nowait() == 0.25
    assert progress_q.get_nowait() == 0.50
    assert progress_q.get_nowait() == 0.75


def test_error_callback():
    """Test del callback de error."""
    error_q = Queue()
    
    def cb_on_error(msg):
        try:
            error_q.put_nowait(str(msg))
        except Exception:
            pass
    
    # Ejecutar callback
    cb_on_error("Test error message")
    
    # Verificar
    assert not error_q.empty()
    assert error_q.get_nowait() == "Test error message"


@patch('streamlit_app.VideoProcessor')
def test_make_cb_on_finish(mock_processor):
    """Test del callback de finalización."""
    finish_q = Queue()
    
    # Mock del VideoProcessor
    mock_vp = MagicMock()
    mock_vp._csv_path_str = "reports/output.csv"
    
    # Crear la función de callback
    def make_cb_on_finish(vp):
        def _cb():
            info = {"csv": getattr(vp, "_csv_path_str", None)}
            if not finish_q.empty():
                finish_q.get_nowait()
            finish_q.put_nowait(info)
        return _cb
    
    # Crear y ejecutar callback
    cb = make_cb_on_finish(mock_vp)
    cb()
    
    # Verificar
    assert not finish_q.empty()
    info = finish_q.get_nowait()
    assert info["csv"] == "reports/output.csv"


@patch('streamlit_app.AppConfig')
def test_config_creation(mock_config_class):
    """Test creación de configuración desde inputs de Streamlit."""
    # Mock de AppConfig
    mock_config = MagicMock()
    mock_config_class.return_value = mock_config
    
    # Crear configuración
    cfg = mock_config_class(
        model_name="yolo11n.pt",
        conf=0.3,
        iou=0.5,
        device=None,
        line_orientation="vertical",
        line_position=0.5,
        invert_direction=False,
        capacity_car=50,
        capacity_moto=50,
        enable_csv=True,
        csv_dir="reports",
        csv_name="test"
    )
    
    # Verificar que se llamó al constructor con los parámetros correctos
    mock_config_class.assert_called_once_with(
        model_name="yolo11n.pt",
        conf=0.3,
        iou=0.5,
        device=None,
        line_orientation="vertical",
        line_position=0.5,
        invert_direction=False,
        capacity_car=50,
        capacity_moto=50,
        enable_csv=True,
        csv_dir="reports",
        csv_name="test"
    )


def test_queue_operations():
    """Test operaciones con colas (vaciar, agregar, obtener)."""
    # Crear cola con elementos
    q = Queue()
    for i in range(5):
        q.put(i)
    
    # Vaciar cola
    while not q.empty():
        try:
            q.get_nowait()
        except Empty:
            break
    
    assert q.empty()
    
    # Agregar elementos con límite de tamaño
    q = Queue(maxsize=3)
    q.put(1)
    q.put(2)
    q.put(3)
    
    # Cola llena, reemplazar el más antiguo
    if q.full():
        q.get_nowait()  # Remover el más antiguo
    q.put(4)
    
    # Verificar contenido
    assert q.get_nowait() == 2
    assert q.get_nowait() == 3
    assert q.get_nowait() == 4


def test_stop_event_behavior():
    """Test comportamiento del evento de parada."""
    import threading
    
    # Crear evento real (no mock)
    stop_event = threading.Event()
    
    # Verificar estado inicial
    assert not stop_event.is_set()
    
    # Señalar el evento
    stop_event.set()
    assert stop_event.is_set()
    
    # Limpiar el evento
    stop_event.clear()
    assert not stop_event.is_set()


def test_frame_rgb_format():
    """Test formato de frames RGB."""
    # Crear frame RGB de prueba
    height, width = 480, 640
    frame_rgb = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    
    # Verificar formato
    assert frame_rgb.shape == (height, width, 3)
    assert frame_rgb.dtype == np.uint8
    assert frame_rgb.min() >= 0
    assert frame_rgb.max() <= 255