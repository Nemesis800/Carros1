from __future__ import annotations
import os
import sys
import argparse
import threading

# Importamos la configuración central y el procesador principal
from config import AppConfig
from processor import VideoProcessor


def parse_cli_args(argv: list[str]) -> argparse.Namespace:
    """
    Define y parsea los argumentos disponibles en modo CLI.
    Retorna un objeto Namespace con todos los parámetros.
    """
    p = argparse.ArgumentParser(description="Conteo de vehículos (headless/CLI)")

    # Argumentos principales (fuente de video)
    p.add_argument("--cli", action="store_true", help="Usar modo CLI (sin UI)")
    src_grp = p.add_mutually_exclusive_group()
    src_grp.add_argument("--source", type=str, help="Ruta a video (mp4/avi/...)", default=None)
    src_grp.add_argument("--webcam", action="store_true", help="Usar webcam (ID 0)")

    # Configuración del modelo y detección
    p.add_argument("--model", type=str, default="yolo11n.pt",
                   help="Modelo YOLO (yolo12n.pt|yolo11n.pt|yolov8n.pt)")
    p.add_argument("--conf", type=float, default=0.3, help="Confianza (0.1-0.8)")

    # Configuración de línea de conteo
    p.add_argument("--orientation", choices=["horizontal", "vertical"], default="vertical",
                   help="Orientación de línea")
    p.add_argument("--line-pos", type=float, default=0.5, help="Posición de línea [0.1-0.9]")
    p.add_argument("--invert", action="store_true", help="Invertir dirección IN/OUT")

    # Capacidades por tipo de vehículo
    p.add_argument("--cap-car", type=int, default=50, help="Capacidad carros")
    p.add_argument("--cap-moto", type=int, default=50, help="Capacidad motos")

    # Configuración de CSV
    p.add_argument("--csv", dest="csv", action="store_true", help="Guardar CSV de eventos")
    p.add_argument("--no-csv", dest="csv", action="store_false", help="No guardar CSV")
    p.set_defaults(csv=True)
    p.add_argument("--csv-dir", type=str, default="reports", help="Carpeta para CSV")
    p.add_argument("--csv-name", type=str, default="",
                   help="Nombre del archivo CSV (opcional, sin ruta)")

    # Mostrar o no ventanas (normalmente en CLI = sin ventanas)
    dis = p.add_mutually_exclusive_group()
    dis.add_argument("--display", action="store_true", help="Mostrar ventanas (no recomendado en CLI)")
    dis.add_argument("--no-display", action="store_true", help="Sin ventanas (headless)")

    return p.parse_args(argv)


def main_cli(ns: argparse.Namespace) -> int:
    """
    Ejecuta el pipeline en modo CLI.
    - Selecciona la fuente (webcam o archivo).
    - Construye AppConfig con los parámetros recibidos.
    - Lanza VideoProcessor en modo sin UI (headless).
    """
    # Fuente de video
    if ns.webcam:
        source = 0
    elif ns.source:
        source = ns.source
        if not os.path.exists(source):
            print(f"[ERROR] Archivo no encontrado: {source}")
            return 2
    else:
        print("[ERROR] Debes indicar --webcam o --source <video>")
        return 2

    # Construimos configuración de la app
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

    # Creamos evento de stop (para cancelar ejecución si se requiere)
    stop_event = threading.Event()

    # Instanciamos el procesador de video
    vp = VideoProcessor(
        video_source=source,
        config=cfg,
        stop_event=stop_event,
        on_error=lambda m: print("[ERROR]", m),
        on_finish=None,
        # display=False por defecto, salvo que se indique --display
        display=bool(ns.display and not ns.no_display) if (ns.display or ns.no_display) else False,
    )

    # Ejecutamos sincronamente (no en hilo separado)
    vp.run()
    return 0
