# services/inference_server.py
"""
Servidor gRPC para el servicio VehicleService.

Este servidor expone RPCs que permiten procesar videos o streams de webcam
usando el modelo YOLO y el pipeline definido en VideoProcessor.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from concurrent import futures
from pathlib import Path
from queue import Empty, Queue
from typing import Iterator, Optional

import cv2
import grpc
import numpy as np

# --- Configuración de paths del proyecto ---
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PROTO = ROOT / "proto"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(PROTO) not in sys.path:
    sys.path.insert(0, str(PROTO))

# --- Imports del proyecto ---
from config import AppConfig  # type: ignore
from processor import VideoProcessor  # type: ignore

# --- Stubs gRPC ---
import vehicle_pb2 as vpb  # type: ignore
import vehicle_pb2_grpc as vpb_grpc  # type: ignore

# --- Configuración de logging ---
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def _cfg_from_msg(msg: vpb.AppConfigMsg) -> AppConfig:
    """Convierte un mensaje gRPC AppConfigMsg en un objeto AppConfig."""
    return AppConfig(
        model_name=msg.model_name or "yolo11n.pt",
        conf=float(msg.conf or 0.3),
        iou=float(msg.iou or 0.5),
        device=msg.device or None,
        line_orientation=msg.line_orientation or "vertical",
        line_position=float(msg.line_position or 0.5),
        invert_direction=bool(msg.invert_direction),
        capacity_car=int(msg.capacity_car or 50),
        capacity_moto=int(msg.capacity_moto or 50),
        enable_csv=bool(msg.enable_csv),
        csv_dir=msg.csv_dir or "reports",
        csv_name=msg.csv_name or "",
    )


def _jpeg_from_rgb(frame_rgb: np.ndarray) -> Optional[bytes]:
    """Convierte un frame RGB en JPEG comprimido.

    Args:
        frame_rgb: Imagen en formato RGB.

    Returns:
        Bytes de imagen JPEG o None si falla.
    """
    try:
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return buf.tobytes() if ok else None
    except Exception:
        return None


class VehicleService(vpb_grpc.VehicleServiceServicer):
    """Implementación del servicio VehicleService definido en vehicle.proto."""

    def _run_pipeline_stream(
        self,
        source: int | str,
        cfg: AppConfig,
        stream_frames: bool,
    ) -> Iterator[vpb.ProcessUpdate]:
        """Ejecuta VideoProcessor y produce un stream de ProcessUpdate.

        Genera:
            - `progress` periódicamente.
            - `frame_jpeg` si stream_frames=True.
            - `csv_path` con done=True al finalizar.
            - `error` si ocurre excepción.
        """
        frame_q: Queue = Queue(maxsize=1)
        prog_q: Queue = Queue(maxsize=16)
        finish_q: Queue = Queue(maxsize=1)
        error_q: Queue = Queue(maxsize=4)

        # Callbacks desde VideoProcessor
        def cb_on_frame(rgb: np.ndarray) -> None:
            if not stream_frames:
                return
            if frame_q.full():
                try:
                    frame_q.get_nowait()
                except Empty:
                    pass
            frame_q.put_nowait(rgb)

        def cb_on_progress(p: float) -> None:
            if prog_q.full():
                try:
                    prog_q.get_nowait()
                except Empty:
                    pass
            prog_q.put_nowait(float(p))

        def cb_on_error(msg: str) -> None:
            error_q.put_nowait(str(msg))

        def cb_on_finish(vp: VideoProcessor):
            def _cb():
                info = getattr(vp, "_csv_path_str", None)
                if not finish_q.empty():
                    try:
                        finish_q.get_nowait()
                    except Empty:
                        pass
                finish_q.put_nowait(info)

            return _cb

        # Lanzamos el worker en un hilo
        import threading

        stop_event = threading.Event()
        vp = VideoProcessor(
            video_source=source,
            config=cfg,
            stop_event=stop_event,
            on_error=cb_on_error,
            on_finish=None,  # se setea abajo con vp
            display=False,
            on_frame=cb_on_frame,
            on_progress=cb_on_progress,
        )
        vp.on_finish = cb_on_finish(vp)
        vp.start()

        # Bucle de publicación de updates
        try:
            last_prog = -1
            while True:
                # Manejo de errores
                try:
                    err = error_q.get_nowait()
                    yield vpb.ProcessUpdate(error=str(err), done=True)
                    break
                except Empty:
                    pass

                # Progreso
                try:
                    p = prog_q.get_nowait()
                    step = int(p * 100)
                    if step != last_prog:
                        last_prog = step
                        yield vpb.ProcessUpdate(progress=float(p))
                except Empty:
                    pass

                # Frames (si aplica)
                if stream_frames:
                    try:
                        rgb = frame_q.get_nowait()
                        jpeg = _jpeg_from_rgb(rgb)
                        if jpeg:
                            yield vpb.ProcessUpdate(frame_jpeg=jpeg)
                    except Empty:
                        pass

                # Finalización
                try:
                    csv_path = finish_q.get_nowait()
                    yield vpb.ProcessUpdate(
                        csv_path=str(csv_path or ""), done=True, progress=1.0
                    )
                    break
                except Empty:
                    pass

                time.sleep(0.1)  # Evita consumir CPU innecesariamente
        finally:
            # Se permite que VideoProcessor cierre de forma natural
            pass

    # --- Métodos RPC ---
    def ProcessVideo(self, request: vpb.ProcessVideoRequest, context):
        """Procesa un archivo de video."""
        cfg = _cfg_from_msg(request.config)
        video_path = request.video_path

        if not video_path or not os.path.exists(video_path):
            yield vpb.ProcessUpdate(error="Archivo de video no encontrado.", done=True)
            return

        yield from self._run_pipeline_stream(video_path, cfg, request.stream_frames)

    def ProcessWebcam(self, request: vpb.ProcessWebcamRequest, context):
        """Procesa un stream de webcam en vivo."""
        cfg = _cfg_from_msg(request.config)
        cam_id = int(request.cam_id or 0)
        yield from self._run_pipeline_stream(cam_id, cfg, request.stream_frames)


def serve(bind_addr: str = "[::]:50051") -> None:
    """Lanza el servidor gRPC."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    vpb_grpc.add_VehicleServiceServicer_to_server(VehicleService(), server)
    server.add_insecure_port(bind_addr)
    LOGGER.info("VehicleService escuchando en %s", bind_addr)
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()

