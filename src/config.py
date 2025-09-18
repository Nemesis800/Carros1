from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

# Nombre de la ventana para la UI (usado por la capa de procesado solo para display)
WINDOW_NAME = "Conteo de Vehículos - YOLOv11/12 + Supervision"


def sanitize_filename(name: str) -> str:
    """
    Reemplaza caracteres no válidos en Windows/macOS/Linux en el nombre de archivo.
    - Sustituye caracteres reservados por guiones bajos.
    - Si el nombre queda vacío, retorna 'reporte'.
    Esto evita errores al crear archivos CSV en distintos sistemas operativos.
    """
    invalid = '<>:"/\\|?*'
    cleaned = "".join(('_' if ch in invalid else ch) for ch in name).strip()
    return cleaned or "reporte"


@dataclass
class AppConfig:
    """
    Estructura de configuración principal de la aplicación.
    Contiene todos los parámetros necesarios para detección, conteo y generación de reportes.
    """

    # Modelo YOLO a usar (con fallback automático en detector.py)
    model_name: str = "yolo11n.pt" 

    # Parámetros de detección
    conf: float = 0.3   
    iou: float = 0.5    
    device: Optional[str] = None 

    # Línea de conteo
    line_orientation: str = "vertical"  
    line_position: float = 0.5           
    invert_direction: bool = False        

    # Capacidades máximas de cada tipo de vehículo
    capacity_car: int = 50
    capacity_moto: int = 50

    # Configuración de reportes CSV
    enable_csv: bool = True            
    csv_dir: str = "resultados"             
    csv_name: Optional[str] = "Registro"
