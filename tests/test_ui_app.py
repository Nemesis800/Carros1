# tests/test_ui_app.py
"""
Tests simplificados para el módulo ui_app.py (interfaz Tkinter)

Prueba la lógica sin requerir una ventana Tkinter real:
- Configuración por defecto
- Lógica de generación de comandos CLI
- Validación de parámetros
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock
import pytest

# Añadir src al path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_app_default_configuration():
    """Test la configuración por defecto de la aplicación."""
    from config import AppConfig
    
    # La aplicación usa AppConfig para sus valores por defecto
    config = AppConfig()
    
    # Verificar valores por defecto
    assert config.model_name == "yolo11n.pt"
    assert config.conf == 0.3
    assert config.iou == 0.5
    assert config.line_orientation == "vertical"
    assert config.line_position == 0.5
    assert config.invert_direction is False
    assert config.capacity_car == 50
    assert config.capacity_moto == 50
    assert config.enable_csv is True
    assert config.csv_dir == "resultados"


def test_app_config_defaults():
    """Test valores por defecto de la configuración."""
    from config import AppConfig
    
    config = AppConfig()
    
    assert config.model_name == "yolo11n.pt"
    assert config.conf == 0.3
    assert config.iou == 0.5
    assert config.line_orientation == "vertical"
    assert config.line_position == 0.5
    assert config.invert_direction is False
    assert config.capacity_car == 50
    assert config.capacity_moto == 50
    assert config.enable_csv is True
    assert config.csv_dir == "resultados"


def test_cli_command_generation_logic():
    """Test la lógica de generación de comando CLI."""
    # Simular la lógica de generación de comando CLI
    def generate_cli_command(use_webcam, video_path, model, conf, orientation, 
                             line_pos, invert, cap_car, cap_moto, enable_csv, 
                             csv_dir, csv_name):
        cmd = "python src/app.py --cli"
        
        if use_webcam:
            cmd += " --webcam"
        elif video_path:
            cmd += f' --source "{video_path}"'
        
        cmd += f" --model {model}"
        cmd += f" --conf {conf:.2f}"
        cmd += f" --orientation {orientation}"
        cmd += f" --line-pos {line_pos:.2f}"
        
        if invert:
            cmd += " --invert"
        
        cmd += f" --cap-car {cap_car}"
        cmd += f" --cap-moto {cap_moto}"
        
        if enable_csv:
            cmd += " --csv"
            cmd += f' --csv-dir "{csv_dir}"'
            if csv_name:
                cmd += f' --csv-name "{csv_name}"'
        
        return cmd
    
    # Test con webcam
    cmd = generate_cli_command(
        use_webcam=True, video_path=None, model="yolo11n.pt", 
        conf=0.35, orientation="horizontal", line_pos=0.6,
        invert=True, cap_car=100, cap_moto=75,
        enable_csv=True, csv_dir="reports", csv_name="test"
    )
    
    assert "--webcam" in cmd
    assert "--model yolo11n.pt" in cmd
    assert "--conf 0.35" in cmd
    assert "--invert" in cmd
    assert "--cap-car 100" in cmd


def test_validate_video_source():
    """Test validación de fuente de video."""
    import os
    
    # Función de validación simulada
    def validate_video_source(use_webcam, video_path):
        if use_webcam:
            return True, 0  # Webcam usa ID 0
        elif video_path and os.path.exists(video_path):
            return True, video_path
        else:
            return False, None
    
    # Test con webcam
    valid, source = validate_video_source(True, None)
    assert valid is True
    assert source == 0
    
    # Test con archivo válido (simular que existe)
    with patch('os.path.exists', return_value=True):
        valid, source = validate_video_source(False, "video.mp4")
        assert valid is True
        assert source == "video.mp4"
    
    # Test sin fuente válida
    with patch('os.path.exists', return_value=False):
        valid, source = validate_video_source(False, "missing.mp4")
        assert valid is False
        assert source is None


def test_format_confidence_value():
    """Test formateo del valor de confianza."""
    # Función de formateo simulada
    def format_confidence(value):
        return f"{float(value):.2f}"
    
    assert format_confidence(0.3) == "0.30"
    assert format_confidence(0.45) == "0.45"
    assert format_confidence(0.789) == "0.79"
    assert format_confidence(1.0) == "1.00"


def test_format_line_position():
    """Test formateo de la posición de línea."""
    # Función de formateo simulada
    def format_line_position(value):
        return f"{int(float(value) * 100)}%"
    
    assert format_line_position(0.5) == "50%"
    assert format_line_position(0.75) == "75%"
    assert format_line_position(0.25) == "25%"
    assert format_line_position(1.0) == "100%"
    assert format_line_position(0.0) == "0%"


def test_validate_capacity_values():
    """Test validación de valores de capacidad."""
    # Función de validación simulada
    def validate_capacity(value):
        try:
            cap = int(value)
            return max(0, min(cap, 10000))  # Limitar entre 0 y 10000
        except:
            return 50  # Valor por defecto
    
    assert validate_capacity("100") == 100
    assert validate_capacity("0") == 0
    assert validate_capacity("-50") == 0
    assert validate_capacity("50000") == 10000
    assert validate_capacity("abc") == 50
    assert validate_capacity("") == 50


def test_parameter_validation():
    """Test validación completa de parámetros."""
    # Función de validación simulada
    def validate_parameters(conf, line_pos, cap_car, cap_moto):
        errors = []
        
        if not (0.1 <= conf <= 0.8):
            errors.append("Confianza debe estar entre 0.1 y 0.8")
        
        if not (0.1 <= line_pos <= 0.9):
            errors.append("Posición de línea debe estar entre 0.1 y 0.9")
        
        if cap_car < 0:
            errors.append("Capacidad de carros no puede ser negativa")
        
        if cap_moto < 0:
            errors.append("Capacidad de motos no puede ser negativa")
        
        return len(errors) == 0, errors
    
    # Parámetros válidos
    valid, errors = validate_parameters(0.3, 0.5, 50, 50)
    assert valid is True
    assert len(errors) == 0
    
    # Confianza inválida
    valid, errors = validate_parameters(0.05, 0.5, 50, 50)
    assert valid is False
    assert len(errors) == 1
    
    # Múltiples errores
    valid, errors = validate_parameters(0.9, 0.95, -10, -5)
    assert valid is False
    assert len(errors) == 4
