# src/cli.py
"""
Interfaz de línea de comandos (CLI) para el sistema de conteo de vehículos.

Permite configurar y ejecutar el pipeline en modo headless (sin UI),
ya sea usando un archivo de video o una webcam como fuente.
"""

from __future__ import annotations

import argparse
import os
import threading
from argparse import Namespace

from config import AppConfig
from processor import VideoProcessor


def parse_cli_args(argv: list[str]) -> Namespace:
    """Define y parsea los argumentos disponibles en modo CLI.

    Args:
        argv: Lista de argumentos de entrada (ej. sys.argv[1:]).

    Returns:
        Objeto Namespace con los parámetros parseados.
    """
    parser = argparse.ArgumentParser(description="Conteo de vehículos (modo CLI)")

    # Fuente de video (mutuamente excluyentes)
    parser.add_argument("--cli", action="store_true", help="Forzar ejecución en modo CLI")
    src_grp = parser.add_mutually_exclusive_group()
    src_grp.add_argument("--source", type=str, default=None, help="Ruta al video (mp4/avi/...)")
    src_grp.add_argument("--webcam", action="store_true", help="Usar webcam (ID 0 por defecto)")

    # Configuración del modelo
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Modelo YOLO a usar (ej. yolo12n.pt | yolo11n.pt | yolov8n.pt)",
    )
    parser.add_argument("--conf", type=float, default=0.3, help="Confianza de detección (0.1–0.8)")

    # Línea de conteo
    parser.add_argument(
        "--orientation",
        choices=["horizontal", "vertical"],
        default="vertical",
        help="Orientación de la línea de conteo",
    )
    parser.add_argument(
        "--line-pos",
        type=float,
        default=0.5,
        help="Posición de la línea [0.1–0.9]",
    )
    parser.add_argument("--invert", action="store_true", help="Invertir dirección IN/OUT")

    # Capacidades de parqueo
    parser.add_argument("--cap-car", type=int, default=50, help="Capacidad para carros")
    parser.add_argument("--cap-moto", type=int, default=50, help="Capacidad para motos")

    # CSV de salida
    parser.add_argument("--csv", dest="csv", action="store_true", help="Guardar CSV de eventos")
    parser.add_argument("--no-csv", dest="csv", action="store_false", help="No guardar CSV")
    parser.set_defaults(csv=True)
    parser.add_argument("--csv-dir", type=str, default="reports", help="Carpeta destino para CSV")
    parser.add_argument(
        "--csv-name",
        type=str,
        default="",
        help="Nombre del archivo CSV (sin ruta, opcional)",
    )

    # Display de ventanas
    dis = parser.add_mutually_exclusive_group()
    dis.add_argument("--display", action="store_true", help="Mostrar ventanas (no recomendado en CLI)")
    dis.add_argument("--no-display", action="store_true", help="Ejecutar sin ventanas (headless)")

    return parser.parse_args(argv)


def main_cli(ns: Namespace) -> int:
    """Ejecuta el pipeline de conteo de vehículos en modo CLI.

    Args:
        ns: Argumentos ya parseados con `parse_cli_args`.

    Returns:
        Código de salida (0 = éxito, 2 = error de entrada).
    """
    # --- Selección de fuente ---
    if ns.webcam:
        source: int | str = 0
    elif ns.source:
        source = ns.source
        if not os.path.exists(source):
            print(f"[ERROR] Archivo no encontrado: {source}")
            return 2
    else:
        print("[ERROR] Debes indicar --webcam o --source <video>")
        return 2

    # --- Configuración de la aplicación ---
    cfg = AppConfig(
        model_name=ns.model,
        conf=float(ns.conf),
        iou=0.5,
        device=None,
        line_orientation=ns.orientation,
        line_position=float(ns.line_pos),
        invert_direction=bool(ns.invert),
        capacity_car=int(ns.cap_car),
        capacity_moto=int(ns.cap_moto),
        enable_csv=bool(ns.csv),
        csv_dir=str(ns.csv_dir),
        csv_name=str(ns.csv_name or ""),
    )

    # --- Control de ejecución ---
    stop_event = threading.Event()

    # Procesador de video (headless por defecto)
    vp = VideoProcessor(
        video_source=source,
        config=cfg,
        stop_event=stop_event,
        on_error=lambda msg: print("[ERROR]", msg),
        on_finish=None,
        display=(
            bool(ns.display and not ns.no_display)
            if (ns.display or ns.no_display)
            else False
        ),
    )

    # Ejecutar de forma sincrónica
    vp.run()
    return 0
