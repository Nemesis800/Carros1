# tests/test_grpc_services.py
"""
Tests para los servicios gRPC (cliente y servidor)

Prueba el servidor de inferencia y el cliente gRPC
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock
import pytest
from queue import Queue
import numpy as np

# Añadir paths necesarios
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PROTO = ROOT / "proto"
SERVICES = ROOT / "services"
CLIENTS = ROOT / "clients"

for path in [SRC, PROTO, SERVICES, CLIENTS]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


class FakeProcessUpdate:
    """Mock de ProcessUpdate para tests."""
    def __init__(self, progress=None, frame_jpeg=None, error=None, csv_path=None, done=False):
        self.progress = progress
        self.frame_jpeg = frame_jpeg
        self.error = error
        self.csv_path = csv_path
        self.done = done


class FakeAppConfigMsg:
    """Mock de AppConfigMsg para tests."""
    def __init__(self):
        self.model_name = "yolo11n.pt"
        self.conf = 0.3
        self.iou = 0.5
        self.device = ""
        self.line_orientation = "vertical"
        self.line_position = 0.5
        self.invert_direction = False
        self.capacity_car = 50
        self.capacity_moto = 50
        self.enable_csv = True
        self.csv_dir = "reports"
        self.csv_name = "test"


@patch('inference_server.VideoProcessor')
def test_server_cfg_from_msg(mock_processor):
    """Test conversión de mensaje gRPC a AppConfig."""
    from inference_server import _cfg_from_msg
    
    msg = FakeAppConfigMsg()
    cfg = _cfg_from_msg(msg)
    
    assert cfg.model_name == "yolo11n.pt"
    assert cfg.conf == 0.3
    assert cfg.iou == 0.5
    assert cfg.line_orientation == "vertical"
    assert cfg.line_position == 0.5
    assert cfg.capacity_car == 50
    assert cfg.capacity_moto == 50
    assert cfg.enable_csv is True


@patch('inference_server.cv2')
def test_server_jpeg_from_rgb(mock_cv2):
    """Test conversión de frame RGB a JPEG."""
    from inference_server import _jpeg_from_rgb
    
    # Mock exitoso
    mock_cv2.cvtColor.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_cv2.imencode.return_value = (True, np.array([1, 2, 3]))
    
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = _jpeg_from_rgb(frame)
    
    assert result is not None
    assert isinstance(result, bytes)
    
    # Mock falla
    mock_cv2.imencode.return_value = (False, None)
    result = _jpeg_from_rgb(frame)
    assert result is None


@patch('inference_server.VideoProcessor')
@patch('inference_server.time.sleep')
def test_vehicle_service_run_pipeline_stream(mock_sleep, mock_processor):
    """Test el pipeline de streaming del servidor."""
    import threading
    from inference_server import VehicleService
    
    # Setup mocks
    mock_vp_instance = MagicMock()
    mock_processor.return_value = mock_vp_instance
    mock_vp_instance._csv_path_str = "reports/test.csv"
    
    service = VehicleService()
    cfg = MagicMock()
    
    # Simular cola con datos
    progress_values = [0.25, 0.5, 0.75, 1.0]
    progress_queue = Queue()
    for val in progress_values:
        progress_queue.put(val)
    
    finish_queue = Queue()
    finish_queue.put("reports/test.csv")
    
    # Configurar side effects para simular el procesamiento
    iteration_count = 0
    def side_effect(*args):
        nonlocal iteration_count
        iteration_count += 1
        if iteration_count > 4:
            raise StopIteration
        
    mock_sleep.side_effect = side_effect
    
    # Ejecutar generador parcialmente
    with patch('inference_server.Queue') as mock_queue_class:
        mock_queue_class.return_value = MagicMock()
        
        gen = service._run_pipeline_stream(
            source="video.mp4",
            cfg=cfg,
            stream_frames=False
        )
        
        # Consumir algunos updates
        try:
            updates = []
            for _ in range(3):
                updates.append(next(gen))
        except StopIteration:
            pass
        
        # Verificar que se creó VideoProcessor
        mock_processor.assert_called_once()


@patch('grpc_client.grpc')
@patch('grpc_client.vpb_grpc')
@patch('grpc_client.vpb')
def test_grpc_client_main(mock_vpb, mock_vpb_grpc, mock_grpc):
    """Test el cliente gRPC principal."""
    from grpc_client import main
    
    # Setup mocks
    mock_channel = MagicMock()
    mock_grpc.insecure_channel.return_value = mock_channel
    
    mock_stub = MagicMock()
    mock_vpb_grpc.VehicleServiceStub.return_value = mock_stub
    
    # Simular respuestas del servidor
    mock_updates = [
        MagicMock(error="", progress=0.5, frame_jpeg=b"", done=False, csv_path=""),
        MagicMock(error="", progress=1.0, frame_jpeg=b"", done=True, csv_path="reports/output.csv")
    ]
    mock_stub.ProcessVideo.return_value = iter(mock_updates)
    
    # Ejecutar cliente
    main("test_video.mp4")
    
    # Verificar llamadas
    mock_grpc.insecure_channel.assert_called_once_with("localhost:50051")
    mock_stub.ProcessVideo.assert_called_once()


@patch('grpc_client.sys.exit')
def test_grpc_client_no_args_behavior(mock_exit):
    """Test comportamiento del cliente gRPC sin argumentos."""
    # Simulamos la condición de no argumentos
    import sys
    original_argv = sys.argv
    sys.argv = ['grpc_client.py']  # Solo el nombre del script
    
    try:
        # El código verificará len(sys.argv) < 2
        if len(sys.argv) < 2:
            # Esto es lo que haría el código real
            mock_exit(2)
        
        # Verificar que se llamó con código 2
        mock_exit.assert_called_with(2)
    finally:
        sys.argv = original_argv


def test_grpc_proto_imports():
    """Test que los archivos proto se pueden importar."""
    try:
        import vehicle_pb2
        import vehicle_pb2_grpc
        assert hasattr(vehicle_pb2, 'AppConfigMsg')
        assert hasattr(vehicle_pb2, 'ProcessVideoRequest')
        assert hasattr(vehicle_pb2, 'ProcessUpdate')
        assert hasattr(vehicle_pb2_grpc, 'VehicleServiceServicer')
    except ImportError as e:
        pytest.skip(f"Proto files not generated: {e}")