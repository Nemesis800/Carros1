# tests/test_mlflow_integration.py
"""
Tests para el módulo mlflow_integration.py

Prueba el tracker de MLflow con mocks
"""

import sys
import json
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import pytest

# Añadir src al path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@patch('mlflow_integration.mlflow')
@patch('mlflow_integration.Path')
def test_tracker_initialization(mock_path, mock_mlflow):
    """Test inicialización del tracker de MLflow."""
    # Setup mocks
    mock_path.return_value.resolve.return_value.parents.__getitem__.return_value = Path("/fake/root")
    mock_mlruns_dir = MagicMock()
    mock_mlruns_dir.as_uri.return_value = "file:///fake/root/mlruns"
    mock_path.return_value.resolve.return_value = mock_mlruns_dir
    
    from mlflow_integration import VehicleDetectionMLflowTracker
    
    # Crear tracker
    tracker = VehicleDetectionMLflowTracker(
        experiment_name="test_experiment",
        tracking_subdir="mlruns"
    )
    
    # Verificar llamadas
    mock_mlflow.set_tracking_uri.assert_called_once()
    mock_mlflow.set_experiment.assert_called_once_with("test_experiment")
    assert tracker.experiment_name == "test_experiment"
    assert tracker.run_id is None
    assert tracker.total_detections == 0


@patch('mlflow_integration.mlflow')
@patch('mlflow_integration.time')
def test_start_experiment_run(mock_time, mock_mlflow):
    """Test inicio de un experimento."""
    from mlflow_integration import VehicleDetectionMLflowTracker
    
    # Setup mocks
    mock_time.time.return_value = 1234567890.0
    mock_run = MagicMock()
    mock_run.info.run_id = "test_run_123"
    mock_mlflow.start_run.return_value = mock_run
    
    # Crear tracker e iniciar run
    tracker = VehicleDetectionMLflowTracker()
    
    config = MagicMock()
    config.model_name = "yolo11n.pt"
    config.conf = 0.3
    config.iou = 0.5
    config.device = None
    config.line_orientation = "vertical"
    config.line_position = 0.5
    config.invert_direction = False
    config.capacity_car = 50
    config.capacity_moto = 50
    config.enable_csv = True
    config.csv_dir = "reports"
    
    run_id = tracker.start_experiment_run(
        config=config,
        video_source="video.mp4",
        tags={"custom_tag": "value"}
    )
    
    # Verificar
    assert run_id == "test_run_123"
    assert tracker.run_id == "test_run_123"
    assert tracker.start_time == 1234567890.0
    mock_mlflow.start_run.assert_called_once()
    mock_mlflow.log_params.assert_called()


@patch('mlflow_integration.mlflow')
@patch('mlflow_integration.time')
def test_end_experiment_run(mock_time, mock_mlflow):
    """Test finalización de un experimento."""
    from mlflow_integration import VehicleDetectionMLflowTracker
    
    # Setup
    mock_time.time.side_effect = [1000.0, 1100.0]  # start y end
    tracker = VehicleDetectionMLflowTracker()
    tracker.run_id = "test_run"
    tracker.start_time = 1000.0
    
    # Finalizar run
    tracker.end_experiment_run(status="FINISHED")
    
    # Verificar
    # El tiempo real será time.time() - start_time, que en este caso es 0 porque el mock devuelve lo mismo
    mock_mlflow.log_metric.assert_called_with("total_experiment_duration_seconds", 0.0)
    mock_mlflow.end_run.assert_called_with(status="FINISHED")
    assert tracker.run_id is None
    assert tracker.start_time is None


@patch('mlflow_integration.mlflow')
def test_log_detection_metrics(mock_mlflow):
    """Test registro de métricas de detección."""
    from mlflow_integration import VehicleDetectionMLflowTracker
    
    tracker = VehicleDetectionMLflowTracker()
    tracker.run_id = "test_run"
    tracker.start_time = 1000.0
    
    # Registrar métricas
    tracker.log_detection_metrics(
        frame_detections=5,
        car_count=3,
        moto_count=2,
        fps=30.5,
        processing_time=0.033
    )
    
    # Verificar que se registraron las métricas
    assert tracker.total_detections == 5
    assert tracker.total_frames_processed == 1
    assert len(tracker.fps_samples) == 1
    assert tracker.fps_samples[0] == 30.5
    
    # Verificar llamada a MLflow
    mock_mlflow.log_metrics.assert_called_once()
    metrics = mock_mlflow.log_metrics.call_args[0][0]
    assert "detections_per_frame" in metrics
    assert metrics["cars_detected"] == 3.0
    assert metrics["motorcycles_detected"] == 2.0
    assert metrics["current_fps"] == 30.5


@patch('mlflow_integration.mlflow')
def test_log_counting_events(mock_mlflow):
    """Test registro de eventos de conteo."""
    from mlflow_integration import VehicleDetectionMLflowTracker
    
    tracker = VehicleDetectionMLflowTracker()
    tracker.run_id = "test_run"
    tracker.start_time = 1000.0
    
    # Registrar eventos de conteo
    tracker.log_counting_events(
        car_in=10,
        car_out=5,
        car_inventory=5,
        moto_in=3,
        moto_out=1,
        moto_inventory=2,
        capacity_exceeded=True
    )
    
    # Verificar llamada
    mock_mlflow.log_metrics.assert_called_once()
    metrics = mock_mlflow.log_metrics.call_args[0][0]
    assert metrics["cars_entered_total"] == 10
    assert metrics["cars_exited_total"] == 5
    assert metrics["cars_current_inventory"] == 5
    # La métrica se llama capacity_exceeded, no capacity_alert
    assert metrics["capacity_exceeded"] == 1


@patch('mlflow_integration.mlflow')
@patch('mlflow_integration.Path')
def test_log_model_metadata(mock_path, mock_mlflow):
    """Test registro de metadata del modelo."""
    from mlflow_integration import VehicleDetectionMLflowTracker
    
    # Mock para escritura de archivo
    mock_file = MagicMock()
    mock_path.return_value = mock_file
    mock_file.exists.return_value = False
    
    tracker = VehicleDetectionMLflowTracker()
    tracker.run_id = "test_run"
    
    # Mock del modelo YOLO
    mock_model = MagicMock()
    mock_model.names = {0: "person", 1: "car", 2: "motorcycle"}
    
    # Registrar metadata
    tracker.log_model_metadata(mock_model)
    
    # Verificar
    mock_mlflow.log_params.assert_called_once()
    params = mock_mlflow.log_params.call_args[0][0]
    assert params["model_type"] == "YOLO"
    assert params["num_classes"] == 3
    
    # Verificar que se intentó guardar el JSON
    mock_file.write_text.assert_called_once()


@patch('mlflow_integration.mlflow')
def test_log_system_performance(mock_mlflow):
    """Test registro de rendimiento del sistema."""
    from mlflow_integration import VehicleDetectionMLflowTracker
    
    tracker = VehicleDetectionMLflowTracker()
    tracker.run_id = "test_run"
    tracker.total_frames_processed = 100
    tracker.total_detections = 500
    tracker.fps_samples = [25.0, 30.0, 28.0]
    
    # Registrar rendimiento del sistema (método que sí existe)
    tracker.log_system_performance(
        total_processing_time=10.0,
        total_frames=100,
        memory_usage_mb=256.5
    )
    
    # Verificar métricas
    mock_mlflow.log_metrics.assert_called()
    metrics = mock_mlflow.log_metrics.call_args[0][0]
    assert metrics["total_processing_time_seconds"] == 10.0
    assert metrics["total_frames_processed"] == 100
    assert metrics["average_fps"] == 10.0  # 100 frames / 10 segundos
    assert metrics["peak_memory_usage_mb"] == 256.5
