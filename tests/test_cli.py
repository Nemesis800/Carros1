# tests/test_cli.py
"""
Tests para el módulo cli.py

Prueba:
- Parsing de argumentos de línea de comandos
- Configuración correcta de AppConfig desde argumentos
- Validación de argumentos
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

# Añadir src al path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cli import parse_cli_args, main_cli
from config import AppConfig


def test_parse_cli_args_with_webcam():
    """Test parsing de argumentos con webcam."""
    args = parse_cli_args([
        "--cli",
        "--webcam",
        "--model", "yolo11n.pt",
        "--conf", "0.45",
        "--orientation", "horizontal",
        "--line-pos", "0.75",
        "--cap-car", "100",
        "--cap-moto", "50",
        "--csv",
        "--csv-dir", "reports",
        "--csv-name", "test_report"
    ])
    
    assert args.cli is True
    assert args.webcam is True
    assert args.model == "yolo11n.pt"
    assert args.conf == 0.45
    assert args.orientation == "horizontal"
    assert args.line_pos == 0.75
    assert args.cap_car == 100
    assert args.cap_moto == 50
    assert args.csv is True
    assert args.csv_dir == "reports"
    assert args.csv_name == "test_report"


def test_parse_cli_args_with_video_source():
    """Test parsing de argumentos con archivo de video."""
    args = parse_cli_args([
        "--cli",
        "--source", "video.mp4",
        "--model", "yolov8n.pt",
        "--invert",
        "--no-display"
    ])
    
    assert args.cli is True
    assert args.source == "video.mp4"
    assert args.model == "yolov8n.pt"
    assert args.invert is True
    assert args.no_display is True
    assert args.webcam is False


def test_parse_cli_args_defaults():
    """Test valores por defecto de argumentos."""
    args = parse_cli_args(["--cli", "--webcam"])
    
    assert args.model == "yolo11n.pt"
    assert args.conf == 0.30
    # iou no se expone en CLI, es fijo en 0.5
    assert args.orientation == "vertical"
    assert args.line_pos == 0.50
    assert args.cap_car == 50
    assert args.cap_moto == 50
    assert args.invert is False
    assert args.csv is True  # Por defecto es True según el código
    assert args.no_display is False


@patch('cli.VideoProcessor')
@patch('cli.threading.Event')
def test_main_cli_with_webcam(mock_event, mock_processor):
    """Test ejecución principal con webcam."""
    # Setup mocks
    mock_stop_event = MagicMock()
    mock_event.return_value = mock_stop_event
    mock_vp_instance = MagicMock()
    mock_processor.return_value = mock_vp_instance
    
    # Crear argumentos simulados
    args = MagicMock()
    args.webcam = True
    args.source = None
    args.model = "yolo11n.pt"
    args.conf = 0.3
    args.iou = 0.5
    args.device = None
    args.orientation = "vertical"
    args.line_pos = 0.5
    args.invert = False
    args.cap_car = 50
    args.cap_moto = 50
    args.csv = True
    args.csv_dir = "reports"
    args.csv_name = "test"
    args.display = False  # No tiene display explícito
    args.no_display = False  # No tiene no_display explícito
    
    # Ejecutar
    result = main_cli(args)
    
    # Verificar
    assert result == 0
    mock_processor.assert_called_once()
    # En CLI, se llama run() directamente, no start/join
    mock_vp_instance.run.assert_called_once()
    
    # Verificar que se usó webcam (source=0)
    call_args = mock_processor.call_args[1]
    assert call_args['video_source'] == 0
    # El display es False porque ni display ni no_display están configurados
    # y el default en CLI es headless (False)
    assert call_args['display'] is False


@patch('cli.VideoProcessor')
@patch('cli.os.path.exists')
def test_main_cli_with_invalid_video(mock_exists, mock_processor):
    """Test cuando el archivo de video no existe."""
    mock_exists.return_value = False
    
    args = MagicMock()
    args.webcam = False
    args.source = "nonexistent.mp4"
    args.model = "yolo11n.pt"
    args.conf = 0.3
    args.iou = 0.5
    args.device = None
    args.orientation = "vertical"
    args.line_pos = 0.5
    args.invert = False
    args.cap_car = 50
    args.cap_moto = 50
    args.csv = False
    args.csv_dir = "reports"
    args.csv_name = None
    args.no_display = False
    
    # Debe retornar error código 2 (no código 1)
    result = main_cli(args)
    assert result == 2
    mock_processor.assert_not_called()


@patch('cli.VideoProcessor')
def test_main_cli_headless_mode(mock_processor):
    """Test modo headless (sin display)."""
    mock_vp_instance = MagicMock()
    mock_processor.return_value = mock_vp_instance
    
    args = MagicMock()
    args.webcam = True
    args.source = None
    args.model = "yolo11n.pt"
    args.conf = 0.3
    args.iou = 0.5
    args.device = None
    args.orientation = "horizontal"
    args.line_pos = 0.75
    args.invert = True
    args.cap_car = 100
    args.cap_moto = 100
    args.csv = True
    args.csv_dir = "reports"
    args.csv_name = "headless_test"
    args.no_display = True  # Modo headless
    
    result = main_cli(args)
    
    assert result == 0
    call_args = mock_processor.call_args[1]
    assert call_args['display'] is False  # Sin display
    # enable_mlflow no es un parámetro en la implementación actual del CLI
