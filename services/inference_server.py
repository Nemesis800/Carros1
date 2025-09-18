# services/inference_server.py
from __future__ import annotations
import os
import sys
import time
from pathlib import Path
from queue import Queue, Empty
from typing import Iterator, Optional

import cv2
import numpy as np
import grpc
from concurrent import futures

# --- imports del proyecto ---
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PROTO = ROOT / "proto"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(PROTO) not in sys.path:
    sys.path.insert(0, str(PROTO))

from config import AppConfig  # type: ignore
from processor import VideoProcessor  # type: ignore

# --- stubs gRPC ---
import vehicle_pb2 as vpb  # type: ignore
import vehicle_pb2_grpc as vpb_grpc  # type: ignore


def _cfg_from_msg(msg: vpb.AppConfigMsg) -> AppConfig:
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
    try:
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return None
        return buf.tobytes()
    except Exception:
        return None


class VehicleService(vpb_grpc.VehicleServiceServicer):
    def _run_pipeline_stream(
        self,
        source: int | str,
        cfg: AppConfig,
        stream_frames: bool,
    ) -> Iterator[vpb.ProcessUpdate]:
        """
        Ejecuta VideoProcessor en un hilo y produce un stream de ProcessUpdate:
         - progress periódicamente
         - frames (JPEG) si stream_frames=True
         - csv_path y done=True al finalizar
         - error si ocurre excepción
        """
        frame_q: Queue = Queue(maxsize=1)
        prog_q: Queue = Queue(maxsize=16)
        finish_q: Queue = Queue(maxsize=1)
        error_q: Queue = Queue(maxsize=4)

        def cb_on_frame(rgb: np.ndarray):
            try:
                if not stream_frames:
                    return
                if frame_q.full():
                    try: frame_q.get_nowait()
                    except Empty: pass
                frame_q.put_nowait(rgb)
            except Exception:
                pass

        def cb_on_progress(p: float):
            try:
                if prog_q.full():
                    try: prog_q.get_nowait()
                    except Empty: pass
                prog_q.put_nowait(float(p))
            except Exception:
                pass

        def cb_on_error(msg: str):
            try:
                error_q.put_nowait(str(msg))
            except Exception:
                pass

        def cb_on_finish(vp: VideoProcessor):
            def _cb():
                info = getattr(vp, "_csv_path_str", None)
                try:
                    if not finish_q.empty():
                        try: finish_q.get_nowait()
                        except Empty: pass
                    finish_q.put_nowait(info)
                except Exception:
                    pass
            return _cb

        # Lanzamos el worker
        import threading
        stop_event = threading.Event()
        vp = VideoProcessor(
            video_source=source,
            config=cfg,
            stop_event=stop_event,
            on_error=cb_on_error,
            on_finish=None,   # lo seteamos debajo para capturar vp
            display=False,
            on_frame=cb_on_frame,
            on_progress=cb_on_progress,
        )
        vp.on_finish = cb_on_finish(vp)
        vp.start()

        # Bucle de publicación
        try:
            last_prog = -1
            while True:
                # Prioridad: errores
                try:
                    err = error_q.get_nowait()
                    yield vpb.ProcessUpdate(error=str(err), done=True)
                    break
                except Empty:
                    pass

                # Progreso (si hay)
                try:
                    p = prog_q.get_nowait()
                    step = int(p * 100)
                    if step != last_prog:
                        last_prog = step
                        yield vpb.ProcessUpdate(progress=float(p))
                except Empty:
                    pass

                # Frames
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
                    yield vpb.ProcessUpdate(csv_path=str(csv_path or ""), done=True, progress=1.0)
                    break
                except Empty:
                    pass

                # Pequeño sleep
                time.sleep(0.1)
        finally:
            # No forzamos stop aquí para permitir que cierre limpio
            pass

    # RPCs
    def ProcessVideo(self, request: vpb.ProcessVideoRequest, context):
        cfg = _cfg_from_msg(request.config)
        video_path = request.video_path
        if not video_path or not os.path.exists(video_path):
            yield vpb.ProcessUpdate(error="Archivo de video no encontrado.", done=True)
            return
        for upd in self._run_pipeline_stream(video_path, cfg, request.stream_frames):
            yield upd

    def ProcessWebcam(self, request: vpb.ProcessWebcamRequest, context):
        cfg = _cfg_from_msg(request.config)
        cam_id = int(request.cam_id or 0)
        for upd in self._run_pipeline_stream(cam_id, cfg, request.stream_frames):
            yield upd


def serve(bind_addr: str = "[::]:50051"):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    vpb_grpc.add_VehicleServiceServicer_to_server(VehicleService(), server)
    server.add_insecure_port(bind_addr)
    print(f"[gRPC] VehicleService escuchando en {bind_addr}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
