# src/processor.py
"""
Módulo principal de procesamiento de video.

Incluye la clase `VideoProcessor`, que:
- Captura video (archivo o webcam).
- Detecta vehículos con YOLO.
- Realiza tracking con ByteTrack.
- Cuenta cruces IN/OUT con `LineCrossingCounterByClass`.
- Genera reportes CSV (opcional).
- Integra métricas con MLflow (opcional).
"""

from __future__ import annotations

import csv
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import supervision as sv

# MLflow opcional
try:
    import mlflow
    import mlflow.pytorch

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("⚠️ MLflow no está instalado. Las métricas no se registrarán.")

from config import AppConfig, WINDOW_NAME, sanitize_filename
from counter import LineCrossingCounterByClass
from detector import VehicleDetector
from utils import winsound_beep


class VideoProcessor(threading.Thread):
    """Procesa un stream de video aplicando detección, tracking y conteo."""

    def __init__(
        self,
        video_source: int | str,
        config: AppConfig,
        stop_event: threading.Event,
        on_error: Callable[[str], None] | None = None,
        on_finish: Callable[[], None] | None = None,
        display: bool = True,
        on_frame: Callable[[np.ndarray], None] | None = None,
        on_progress: Callable[[float], None] | None = None,
        enable_mlflow: bool = True,
        experiment_name: str = "vehicle_detection",
        mlflow_tags: dict | None = None,
    ) -> None:
        """Inicializa un procesador de video en un hilo independiente."""
        super().__init__(daemon=True)
        self.video_source = video_source
        self.config = config
        self.stop_event = stop_event
        self.on_error = on_error
        self.on_finish = on_finish
        self.display = display
        self.on_frame = on_frame
        self.on_progress = on_progress

        # MLflow
        self.enable_mlflow = enable_mlflow and MLFLOW_AVAILABLE
        self.experiment_name = experiment_name
        self.mlflow_tags = mlflow_tags or {}
        self.mlflow_run_id = None
        self.mlflow_start_time = None

        # Contadores
        self.frame_count = 0
        self.detection_count = 0
        self.fps_samples = []

        # Estados de alerta de capacidad
        self._prev_over_car = False
        self._prev_over_moto = False

        # Ventana activa (si display=True)
        self.active_window_name = None

        # Referencia al modelo
        self._current_model = None

        # CSV
        self.csv_fp = None
        self.csv_writer = None
        self._csv_path_str = None
        self._prev_counts = {"car_in": 0, "car_out": 0, "moto_in": 0, "moto_out": 0}
        self._last_car_inv = 0
        self._last_moto_inv = 0

    # ------------------------------------------------------------------
    # Utilidades internas
    # ------------------------------------------------------------------
    def _notify_error(self, msg: str) -> None:
        """Notifica un error al callback o lo imprime en consola."""
        if self.enable_mlflow and self.mlflow_run_id:
            try:
                mlflow.log_param("last_error", msg)
            except Exception:
                pass
        if self.on_error:
            try:
                self.on_error(msg)
                return
            except Exception:
                pass
        print(f"[ERROR] {msg}")

    # ------------------------------------------------------------------
    # Inicialización y logging MLflow
    # ------------------------------------------------------------------
    def _init_mlflow(self) -> None:
        """Inicializa MLflow (si está habilitado)."""
        if not self.enable_mlflow:
            return
        try:
            mlflow_dir = Path("mlruns")
            mlflow_dir.mkdir(exist_ok=True)
            mlflow.set_tracking_uri(f"file:///{mlflow_dir.absolute()}")

            try:
                mlflow.create_experiment(self.experiment_name)
            except mlflow.exceptions.MlflowException:
                pass
            mlflow.set_experiment(self.experiment_name)

            run = mlflow.start_run(
                tags={
                    "mlflow.source.type": "LOCAL",
                    "mlflow.source.name": "vehicle_detection_processor",
                    "video_type": "webcam" if isinstance(self.video_source, int) else "file",
                    "timestamp": datetime.now().isoformat(),
                }
            )
            self.mlflow_run_id = run.info.run_id
            self.mlflow_start_time = time.time()

            self._log_config_params()
            print(f"🚀 MLflow Run iniciado: {self.mlflow_run_id}")
        except Exception as e:
            print(f"⚠️ Error inicializando MLflow: {e}")
            self.enable_mlflow = False

    def _log_config_params(self) -> None:
        """Registra los parámetros de AppConfig en MLflow."""
        if not self.enable_mlflow or not self.mlflow_run_id:
            return
        try:
            mlflow.log_params(
                {
                    "model_name": self.config.model_name,
                    "confidence_threshold": self.config.conf,
                    "iou_threshold": self.config.iou,
                    "line_orientation": self.config.line_orientation,
                    "line_position": self.config.line_position,
                    "invert_direction": self.config.invert_direction,
                    "capacity_car": self.config.capacity_car,
                    "capacity_moto": self.config.capacity_moto,
                    "enable_csv": getattr(self.config, "enable_csv", False),
                    "device": self.config.device or "auto",
                    "video_source": str(self.video_source),
                    "display_mode": self.display,
                }
            )
        except Exception as e:
            print(f"⚠️ Error registrando parámetros: {e}")

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------
    def _init_csv(self) -> None:
        """Inicializa el CSV de resultados (si está habilitado)."""
        if not getattr(self.config, "enable_csv", False):
            return
        try:
            Path(self.config.csv_dir).mkdir(parents=True, exist_ok=True)
            ts_name = datetime.now().strftime("%Y%m%d_%H%M%S")
            src_name = "webcam" if isinstance(self.video_source, int) else Path(str(self.video_source)).name

            base = sanitize_filename(str(self.config.csv_name or f"reporte_{ts_name}_{src_name}"))
            filename = base if base.endswith(".csv") else f"{base}.csv"
            csv_path = Path(self.config.csv_dir) / filename

            self.csv_fp = open(csv_path, "w", newline="", encoding="utf-8-sig")
            self.csv_writer = csv.writer(self.csv_fp, delimiter=";")
            self.csv_writer.writerow(
                [
                    "timestamp",
                    "evento",
                    "clase",
                    "car_in",
                    "car_out",
                    "moto_in",
                    "moto_out",
                    "car_inv",
                    "moto_inv",
                    "modelo",
                    "conf",
                    "orientacion",
                    "pos_linea",
                    "invertido",
                    "fuente",
                ]
            )
            self._csv_path_str = str(csv_path)
        except Exception as e:
            self._notify_error(f"No se pudo inicializar CSV: {e}")
    # ------------------------------------------------------------------
    # Línea de conteo (depende del tamaño del frame)
    # ------------------------------------------------------------------
    def _ensure_counter_line(self, frame: np.ndarray) -> "LineCrossingCounterByClass":
        """Crea el contador con la línea ubicada según config y tamaño del frame."""
        h, w = frame.shape[:2]
        pos = float(self.config.line_position)
        pos = max(0.0, min(1.0, pos))

        if self.config.line_orientation == "vertical":
            x = int(pos * w)
            a, b = (x, 0), (x, h)
        else:
            y = int(pos * h)
            a, b = (0, y), (w, y)

        return LineCrossingCounterByClass(
            a=a,
            b=b,
            labels=("car", "motorcycle"),
            invert_direction=bool(self.config.invert_direction),
        )

    # ------------------------------------------------------------------
    # Escritura de CSV: eventos y resumen
    # ------------------------------------------------------------------
    def _write_event_rows(
        self,
        car_in: int,
        car_out: int,
        moto_in: int,
        moto_out: int,
        car_inv: int,
        moto_inv: int,
    ) -> None:
        """Escribe filas de eventos individuales (IN/OUT por clase) en el CSV."""
        if not self.csv_writer:
            return

        ts = datetime.now().isoformat()
        common_tail = [
            self.config.model_name,
            f"{self.config.conf:.2f}",
            self.config.line_orientation,
            f"{self.config.line_position:.2f}",
            str(bool(self.config.invert_direction)),
            str(self.video_source),
        ]

        # Por compatibilidad con tests: escribir una fila por evento unitario
        for _ in range(int(car_in)):
            self.csv_writer.writerow(
                [ts, "IN", "car", car_in, car_out, moto_in, moto_out, car_inv, moto_inv, *common_tail]
            )
        for _ in range(int(car_out)):
            self.csv_writer.writerow(
                [ts, "OUT", "car", car_in, car_out, moto_in, moto_out, car_inv, moto_inv, *common_tail]
            )
        for _ in range(int(moto_in)):
            self.csv_writer.writerow(
                [ts, "IN", "motorcycle", car_in, car_out, moto_in, moto_out, car_inv, moto_inv, *common_tail]
            )
        for _ in range(int(moto_out)):
            self.csv_writer.writerow(
                [ts, "OUT", "motorcycle", car_in, car_out, moto_in, moto_out, car_inv, moto_inv, *common_tail]
            )

        # Actualiza estado previo (para deltas en el loop)
        self._prev_counts["car_in"] += int(car_in)
        self._prev_counts["car_out"] += int(car_out)
        self._prev_counts["moto_in"] += int(moto_in)
        self._prev_counts["moto_out"] += int(moto_out)
        self._last_car_inv = int(car_inv)
        self._last_moto_inv = int(moto_inv)

    def _write_summary(self) -> None:
        """Escribe una fila de resumen al final del CSV."""
        if not self.csv_writer:
            return

        ts = datetime.now().isoformat()
        # El test solo valida que exista ";SUMMARY;-", pero mantenemos todas las columnas
        self.csv_writer.writerow(
            [
                ts,
                "SUMMARY",
                "-",
                self._prev_counts.get("car_in", 0),
                self._prev_counts.get("car_out", 0),
                self._prev_counts.get("moto_in", 0),
                self._prev_counts.get("moto_out", 0),
                self._last_car_inv,
                self._last_moto_inv,
                self.config.model_name,
                f"{self.config.conf:.2f}",
                self.config.line_orientation,
                f"{self.config.line_position:.2f}",
                str(bool(self.config.invert_direction)),
                str(self.video_source),
            ]
        )

    # ------------------------------------------------------------------
    # Loop principal
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Bucle principal: captura → detecta → trackea → cuenta → reporta."""
        cap = None
        counter = None
        try:
            cap = cv2.VideoCapture(self.video_source, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                # Reintento con backend por defecto, por si FFMPEG no está disponible
                cap = cv2.VideoCapture(self.video_source)
            if not cap.isOpened():
                import cv2 as _cv2, os as _os
                self._notify_error(
                    f"No se pudo abrir la fuente: {self.video_source} | "
                    f"existe={_os.path.exists(str(self.video_source))} | "
                    f"opencv={_cv2.__version__}"
                )
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            detector = VehicleDetector(
                model_name=self.config.model_name,
                conf=self.config.conf,
                iou=self.config.iou,
                device=self.config.device,
            )
            tracker = sv.ByteTrack()

            self._init_csv()
            self._init_mlflow()

            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                # Inicializa la línea de conteo usando el primer frame
                if counter is None:
                    counter = self._ensure_counter_line(frame)

                # Inferencia y tracking
                detections = detector.detect(frame)
                tracked = tracker.update_with_detections(detections)
                counter.update(tracked)

                # Deltas respecto al estado previo
                car_in = max(counter.in_counts.get("car", 0) - self._prev_counts["car_in"], 0)
                car_out = max(counter.out_counts.get("car", 0) - self._prev_counts["car_out"], 0)
                moto_in = max(counter.in_counts.get("motorcycle", 0) - self._prev_counts["moto_in"], 0)
                moto_out = max(counter.out_counts.get("motorcycle", 0) - self._prev_counts["moto_out"], 0)

                car_inv = counter.inventory.get("car", 0)
                moto_inv = counter.inventory.get("motorcycle", 0)

                # Escribir eventos si hubo cambios
                if (car_in + car_out + moto_in + moto_out) > 0:
                    self._write_event_rows(car_in, car_out, moto_in, moto_out, car_inv, moto_inv)

                # Mostrar ventana (opcional)
                if self.display:
                    cv2.imshow(WINDOW_NAME, frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            # Guardar resumen al finalizar
            if counter is not None:
                self._last_car_inv = counter.inventory.get("car", 0)
                self._last_moto_inv = counter.inventory.get("motorcycle", 0)
            self._write_summary()

        except Exception as e:
            self._notify_error(f"Fallo en procesamiento: {e}")
        finally:
            if cap:
                cap.release()
            cv2.destroyAllWindows()
            if self.csv_fp:
                try:
                    self.csv_fp.flush()
                except Exception:
                    pass
                self.csv_fp.close()

