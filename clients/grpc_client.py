# clients/grpc_client.py
"""
Cliente gRPC para el servicio VehicleService.

Este script envía una petición al servidor gRPC con la configuración
del modelo YOLO y procesa las respuestas de progreso, errores y resultado final.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import NoReturn

import grpc

# Añadir carpeta proto al sys.path para importar stubs generados
ROOT = Path(__file__).resolve().parents[1]
PROTO = ROOT / "proto"
if str(PROTO) not in sys.path:
    sys.path.insert(0, str(PROTO))

import vehicle_pb2 as vpb  # type: ignore
import vehicle_pb2_grpc as vpb_grpc  # type: ignore

# Configuración básica de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
LOGGER = logging.getLogger(__name__)


def main(video_path: str) -> None:
    """Ejecuta el procesamiento de video en el servidor gRPC.

    Args:
        video_path: Ruta del video a procesar.
    """
    # Conexión con el servidor gRPC en localhost:50051
    chan = grpc.insecure_channel("localhost:50051")
    stub = vpb_grpc.VehicleServiceStub(chan)

    # Mensaje de configuración de la aplicación
    cfg = vpb.AppConfigMsg(
        model_name="yolo11n.pt",
        conf=0.30,
        iou=0.5,
        device="",  # GPU/CPU específico si aplica
        line_orientation="vertical",
        line_position=0.50,
        invert_direction=False,
        capacity_car=50,
        capacity_moto=50,
        enable_csv=True,
        csv_dir=str(ROOT / "reports"),
        csv_name="from_grpc",
    )

    # Solicitud de procesamiento de video
    req = vpb.ProcessVideoRequest(
        video_path=video_path,
        config=cfg,
        stream_frames=False,  # Si es True, el servidor también enviará JPEGs
    )

    # Iterar sobre las respuestas del servidor
    for upd in stub.ProcessVideo(req):
        if upd.error:
            LOGGER.error("Error en el servidor: %s", upd.error)
            break
        if upd.progress:
            LOGGER.info("Progreso: %d%%", int(upd.progress * 100))
        if upd.frame_jpeg:
            LOGGER.debug("Frame recibido: %d bytes", len(upd.frame_jpeg))
        if upd.done:
            LOGGER.info("Procesamiento terminado. CSV: %s", upd.csv_path)
            break


if __name__ == "__main__":
    if len(sys.argv) < 2:
        LOGGER.error("Uso: python clients/grpc_client.py C:\\ruta\\video.mp4")
        sys.exit(2)

    main(sys.argv[1])
