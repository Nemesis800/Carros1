# src/config.py
"""
Módulo de configuración central de la aplicación.

Define la clase `AppConfig`, que concentra todos los parámetros necesarios
para detección, conteo de vehículos y generación de reportes. Incluye también
utilidades relacionadas con configuración (ej. sanitización de nombres).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Nombre de la ventana para UI (usado solo si display=True en VideoProcessor)
WINDOW_NAME = "Conteo de Vehículos - YOLOv11/12 + Supervision"


def sanitize_filename(name: str) -> str:
    """Normaliza un nombre de archivo removiendo caracteres inválidos.

    Reemplaza caracteres reservados en Windows/macOS/Linux por guiones bajos.
    Si el resultado queda vacío, retorna `"reporte"`.

    Args:
        name: Nombre original (posiblemente con caracteres inválidos).

    Returns:
        Nombre seguro para ser usado como archivo CSV.
    """
    invalid = '<>:"/\\|?*'
    cleaned = "".join("_" if ch in invalid else ch for ch in name).strip()
    return cleaned or "reporte"


@dataclass
class AppConfig:
    """Estructura de configuración principal de la aplicación.

    Contiene todos los parámetros necesarios para detección,
    conteo de vehículos y generación de reportes.
    """

    # --- Modelo YOLO ---
    model_name: str = "yolo11n.pt"  # Modelo YOLO por defecto

    # --- Parámetros de detección ---
    conf: float = 0.3  # Umbral de confianza [0–1]
    iou: float = 0.5  # Umbral IoU [0–1]
    device: Optional[str] = None  # CPU o GPU explícita (ej. "cuda:0")

    # --- Línea de conteo ---
    line_orientation: str = "vertical"  # Dirección de la línea ("vertical"/"horizontal")
    line_position: float = 0.5  # Posición relativa [0–1]
    invert_direction: bool = False  # Invierte IN/OUT

    # --- Capacidades máximas ---
    capacity_car: int = 50
    capacity_moto: int = 50

    # --- Configuración de reportes CSV ---
    enable_csv: bool = True
    csv_dir: str = "resultados"  # Carpeta destino para guardar reportes
    csv_name: Optional[str] = "Registro"  # Nombre base del CSV

    # --- Configuración de MLflow ---
    enable_mlflow: bool = False  # Habilitar/deshabilitar integración con MLflow
