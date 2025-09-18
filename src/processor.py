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

    # --------------------------- Utilidades MLflow ----------------------------

    def _ensure_mlflow_dirs(self) -> Path:
        """
        Crea (si no existe) el directorio 'mlruns' en la RAÍZ del proyecto
        (carpeta padre de 'src') y devuelve la ruta absoluta.
        """
        root = Path(__file__).resolve().parents[1]  # repo root
        mlruns_dir = (root / "mlruns").resolve()
        mlruns_dir.mkdir(parents=True, exist_ok=True)
        return mlruns_dir

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
        """Inicializa MLflow (si está habilitado) y asegura la carpeta mlruns en el root."""
        if not self.enable_mlflow:
            return

        # Configurar directorio temporal propio desde el inicio (mitiga permisos en Windows)
        self._setup_temp_directory()

        try:
            mlruns_dir = self._ensure_mlflow_dirs()

            # URI de tracking robusta (Windows friendly)
            try:
                tracking_uri = mlruns_dir.as_uri()  # p.ej. file:///C:/ruta/proyecto/mlruns
            except ValueError:
                tracking_uri = f"file:///{str(mlruns_dir).replace(os.sep, '/')}"

            mlflow.set_tracking_uri(tracking_uri)

            # Crear/seleccionar experimento
            try:
                mlflow.create_experiment(self.experiment_name)
            except Exception:
                pass
            mlflow.set_experiment(self.experiment_name)

            # Iniciar run
            run = mlflow.start_run(
                tags={
                    "mlflow.source.type": "LOCAL",
                    "mlflow.source.name": "vehicle_detection_processor",
                    "video_type": "webcam" if isinstance(self.video_source, int) else "file",
                    "timestamp": datetime.now().isoformat(),
                    **self.mlflow_tags,
                }
            )
            self.mlflow_run_id = run.info.run_id
            self.mlflow_start_time = time.time()

            # Parámetros base
            self._log_config_params()
            print(f"🚀 MLflow Run iniciado: {self.mlflow_run_id}  (tracking: {tracking_uri})")

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
    # MLflow avanzado - logging al final
    # ------------------------------------------------------------------
    def _setup_temp_directory(self) -> None:
        """Configura directorio temporal propio para evitar problemas de permisos."""
        import tempfile

        try:
            app_temp_dir = Path.cwd() / "temp_app"
            app_temp_dir.mkdir(exist_ok=True)

            if not hasattr(self, "_original_temp_config"):
                self._original_temp_config = {
                    "tempdir": tempfile.gettempdir(),
                    "TMP": os.environ.get("TMP"),
                    "TEMP": os.environ.get("TEMP"),
                    "TMPDIR": os.environ.get("TMPDIR"),
                    "HOME": os.environ.get("HOME"),
                    "USERPROFILE": os.environ.get("USERPROFILE"),
                }

            str_temp_dir = str(app_temp_dir.absolute())
            tempfile.tempdir = str_temp_dir
            os.environ["TMP"] = str_temp_dir
            os.environ["TEMP"] = str_temp_dir
            os.environ["TMPDIR"] = str_temp_dir

            # En algunos entornos Windows, MLflow termina usando HOME/USERPROFILE
            current_dir = str(Path.cwd().absolute())
            os.environ["HOME"] = current_dir
            os.environ["USERPROFILE"] = current_dir

            print(f"📁 Directorio temporal configurado: {str_temp_dir}")

        except Exception as e:
            print(f"⚠️ Error configurando directorio temporal: {e}")

    def _restore_temp_directory(self) -> None:
        """Restaura la configuración original del directorio temporal."""
        if not hasattr(self, "_original_temp_config"):
            return

        import tempfile

        try:
            config = self._original_temp_config

            if config["tempdir"]:
                tempfile.tempdir = config["tempdir"]

            for var_name in ["TMP", "TEMP", "TMPDIR", "HOME", "USERPROFILE"]:
                if config[var_name] is not None:
                    os.environ[var_name] = config[var_name]
                elif var_name in os.environ:
                    del os.environ[var_name]

        except Exception as e:
            print(f"⚠️ Error restaurando directorio temporal: {e}")

    def _try_create_visualization(self) -> None:
        """Intenta crear visualización con manejo robusto de errores."""
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Gráfico IN vs OUT
            categories = ["Carros", "Motos"]
            in_counts = [self._prev_counts.get("car_in", 0), self._prev_counts.get("moto_in", 0)]
            out_counts = [self._prev_counts.get("car_out", 0), self._prev_counts.get("moto_out", 0)]

            x_pos = range(len(categories))
            width = 0.35

            ax1.bar([p - width / 2 for p in x_pos], in_counts, width, label="IN")
            ax1.bar([p + width / 2 for p in x_pos], out_counts, width, label="OUT")
            ax1.set_xlabel("Tipo de Vehículo")
            ax1.set_ylabel("Conteo")
            ax1.set_title("Conteo de Vehículos IN vs OUT")
            ax1.set_xticks(list(x_pos))
            ax1.set_xticklabels(categories)
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Gráfico inventario vs capacidad
            inventories = [self._last_car_inv, self._last_moto_inv]
            capacities = [self.config.capacity_car, self.config.capacity_moto]

            ax2.bar(list(x_pos), inventories, alpha=0.7, label="Inventario Actual")
            ax2.bar(list(x_pos), capacities, alpha=0.3, label="Capacidad Máxima")
            ax2.set_xlabel("Tipo de Vehículo")
            ax2.set_ylabel("Cantidad")
            ax2.set_title("Inventario Actual vs Capacidad")
            ax2.set_xticks(list(x_pos))
            ax2.set_xticklabels(categories)
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            # Guardar en temp_app
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_dir = Path.cwd() / "temp_app"
            viz_file = temp_dir / f"visualization_{timestamp}.png"
            fig.savefig(str(viz_file), dpi=150, bbox_inches="tight")
            plt.close(fig)

            # Registrar como artifact (no bloquea si falla)
            try:
                mlflow.log_artifact(str(viz_file), "visualizations")
                print("✅ Visualización creada y guardada en MLflow")
            except Exception as e:
                print(f"⚠️ MLflow artifact error (visualización): {e}")
            finally:
                if viz_file.exists():
                    try:
                        viz_file.unlink()
                    except Exception:
                        pass

        except Exception as e:
            print(f"⚠️ Error creando visualización: {e}")

    def _try_register_csv(self) -> None:
        """Intenta registrar CSV como artifact con manejo robusto de errores."""
        try:
            if self._csv_path_str and Path(self._csv_path_str).exists():
                mlflow.log_artifact(self._csv_path_str, "reports")
                print(f"✅ CSV registrado: {self._csv_path_str}")
        except Exception as e:
            print(f"⚠️ MLflow artifact error (CSV): {e}")

    def _try_register_model(self) -> None:
        """Intenta registrar modelo como artifact con manejo robusto de errores."""
        try:
            model_path = Path(self.config.model_name)
            if model_path.exists():
                mlflow.log_artifact(str(model_path), "model_files")
                print(f"✅ Modelo registrado: {self.config.model_name}")
        except Exception as e:
            print(f"⚠️ MLflow artifact error (modelo): {e}")

    def _apply_advanced_mlflow_features(self) -> None:
        """Aplica características avanzadas de MLflow al finalizar el procesamiento."""
        if not self.enable_mlflow or not self.mlflow_run_id:
            return

        # Asegurar temp y permisos antes de artifacts
        self._setup_temp_directory()

        try:
            import psutil
            import platform

            print("🔍 Aplicando características avanzadas de MLflow...")

            final_metrics = {
                "total_car_in": self._prev_counts.get("car_in", 0),
                "total_car_out": self._prev_counts.get("car_out", 0),
                "total_moto_in": self._prev_counts.get("moto_in", 0),
                "total_moto_out": self._prev_counts.get("moto_out", 0),
                "final_car_inventory": self._last_car_inv,
                "final_moto_inventory": self._last_moto_inv,
                "total_frames_processed": self.frame_count,
                "detection_count": self.detection_count,
            }

            # FPS
            if self.fps_samples:
                final_metrics["avg_fps"] = sum(self.fps_samples) / len(self.fps_samples)
                final_metrics["max_fps"] = max(self.fps_samples)
                final_metrics["min_fps"] = min(self.fps_samples)

            # Tiempo total
            if self.mlflow_start_time:
                processing_time = time.time() - self.mlflow_start_time
                final_metrics["processing_time_seconds"] = processing_time
                if self.frame_count > 0:
                    final_metrics["frames_per_second_avg"] = self.frame_count / processing_time

            mlflow.log_metrics(final_metrics)
            print("✅ Métricas finales registradas")

            # Info de sistema y modelo
            system_info = {
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                "memory_available_gb": psutil.virtual_memory().available / (1024**3),
                "memory_used_percent": psutil.virtual_memory().percent,
                "platform_system": platform.system(),
                "platform_release": platform.release(),
                "python_version": platform.python_version(),
            }

            try:
                import torch

                if torch.cuda.is_available():
                    gpu_props = torch.cuda.get_device_properties(0)
                    system_info.update(
                        {
                            "gpu_name": gpu_props.name,
                            "gpu_memory_total_gb": gpu_props.total_memory / (1024**3),
                            "gpu_compute_capability": f"{gpu_props.major}.{gpu_props.minor}",
                        }
                    )
            except Exception:
                pass

            if self._current_model is not None:
                model_info = {
                    "yolo_model_type": "YOLO",
                    "yolo_model_file": self.config.model_name,
                    "yolo_confidence": self.config.conf,
                    "yolo_iou_threshold": self.config.iou,
                    "yolo_classes": "car,motorcycle",
                    "yolo_device": str(self.config.device) if self.config.device else "auto",
                }
                system_info.update(model_info)

            mlflow.log_params(system_info)
            print("✅ Parámetros del sistema y modelo registrados")

            # Artifacts (no bloquear si falla)
            self._try_create_visualization()
            self._try_register_csv()
            self._try_register_model()

            # Cerrar run
            mlflow.end_run()
            print(f"🏁 MLflow Run finalizado: {self.mlflow_run_id}")
            print("✅ Sistema MLflow avanzado completado exitosamente")

        except Exception as e:
            print(f"⚠️ Error en características avanzadas de MLflow: {e}")
            try:
                mlflow.end_run()
            except Exception:
                pass
        finally:
            self._restore_temp_directory()

    # ------------------------------------------------------------------
    # Loop principal
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Bucle principal: captura → detecta → trackea → cuenta → reporta."""
        cap = None
        counter = None
        detector = None
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

            # Variables para tracking de FPS
            start_time = time.time()
            fps_counter = 0
            fps_update_interval = 30  # Actualizar cada 30 frames

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

                # Actualizar contador de detecciones
                self.detection_count += len(detections)

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

                # ================== RENDER VISUAL (NO TOCAR) ==================

                if self.display or self.on_frame:
                    draw_frame = frame.copy()

                    # 1) Detecciones: cajas + labels (Supervision)
                    box_annotator = sv.BoxAnnotator()
                    label_annotator = sv.LabelAnnotator()

                    labels = []
                    for i in range(len(tracked)):
                        class_name = (
                            tracked.data.get("class_name", [""] * len(tracked))[i]
                            if "class_name" in tracked.data
                            else ""
                        )
                        confidence = tracked.confidence[i] if tracked.confidence is not None else 0.0
                        tracker_id = tracked.tracker_id[i] if tracked.tracker_id is not None else None
                        id_text = f"#{int(tracker_id)}" if tracker_id is not None else ""
                        labels.append(f"{class_name} {id_text} {confidence:.2f}")

                    draw_frame = box_annotator.annotate(scene=draw_frame, detections=tracked)
                    draw_frame = label_annotator.annotate(scene=draw_frame, detections=tracked, labels=labels)

                    # 2) Línea de conteo
                    cv2.line(draw_frame, counter.a, counter.b, (0, 255, 255), 3)
                    cv2.circle(draw_frame, counter.a, 5, (0, 255, 255), -1)
                    cv2.circle(draw_frame, counter.b, 5, (0, 255, 255), -1)

                    # 3) Panel HUD
                    panel_w = 420
                    panel_h = 140
                    overlay = draw_frame.copy()
                    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.7, draw_frame, 0.3, 0, draw_frame)

                    cv2.putText(
                        draw_frame, "Conteo IN/OUT e Inventario", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
                    )

                    car_total_in = counter.in_counts.get("car", 0)
                    car_total_out = counter.out_counts.get("car", 0)
                    cv2.putText(
                        draw_frame,
                        f"Carros -> IN: {car_total_in} | OUT: {car_total_out} | INV: {car_inv}/{self.config.capacity_car}",
                        (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (80, 255, 80),
                        2,
                    )

                    moto_total_in = counter.in_counts.get("motorcycle", 0)
                    moto_total_out = counter.out_counts.get("motorcycle", 0)
                    cv2.putText(
                        draw_frame,
                        f"Motos  -> IN: {moto_total_in} | OUT: {moto_total_out} | INV: {moto_inv}/{self.config.capacity_moto}",
                        (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (80, 200, 255),
                        2,
                    )

                    # Mensaje salir
                    if self.display:
                        exit_msg = "Presiona Q para salir"
                        msg_size = cv2.getTextSize(exit_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        msg_x = draw_frame.shape[1] - msg_size[0] - 15
                        msg_y = 30
                        cv2.rectangle(draw_frame, (msg_x - 5, msg_y - 20), (draw_frame.shape[1] - 5, msg_y + 5), (0, 0, 0), -1)
                        cv2.putText(draw_frame, exit_msg, (msg_x, msg_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    # 4) Alarmas de capacidad (parpadeo)
                    over_car = car_inv > self.config.capacity_car
                    over_moto = moto_inv > self.config.capacity_moto

                    if over_car or over_moto:
                        frame_mod = (self.frame_count // 15) % 2
                        if frame_mod == 0:
                            alert_text = "ALERTA: CAPACIDAD EXCEDIDA"
                            cv2.putText(
                                draw_frame, alert_text, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                            )
                            if over_car:
                                cv2.putText(
                                    draw_frame,
                                    "- CARROS EXCEDIDOS",
                                    (30, 155),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 0, 255),
                                    2,
                                )
                            if over_moto:
                                cv2.putText(
                                    draw_frame,
                                    "- MOTOS EXCEDIDAS",
                                    (30, 175),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 0, 255),
                                    2,
                                )

                    # Beep al entrar en estado excedido
                    if (over_car and not self._prev_over_car) or (over_moto and not self._prev_over_moto):
                        winsound_beep()

                    self._prev_over_car = over_car
                    self._prev_over_moto = over_moto

                    # 5) Mostrar ventana / callback
                    if self.display:
                        if self.active_window_name is None:
                            timestamp = int(time.time() * 1000)
                            self.active_window_name = f"VehicleCounter_{timestamp}"
                            cv2.namedWindow(self.active_window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                            cv2.resizeWindow(
                                self.active_window_name, min(1280, frame.shape[1]), min(720, frame.shape[0])
                            )
                            print(f"🗺️ Ventana OpenCV creada: {self.active_window_name}")

                        cv2.imshow(self.active_window_name, draw_frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            self.stop_event.set()
                            break

                    if self.on_frame:
                        try:
                            rgb_frame = cv2.cvtColor(draw_frame, cv2.COLOR_BGR2RGB)
                            self.on_frame(rgb_frame)
                        except Exception:
                            pass

                # ================== FIN RENDER VISUAL (NO TOCAR) ==============

                # Progreso para UI web
                if self.on_progress and total_frames > 0:
                    progress = min(1.0, self.frame_count / total_frames)
                    if int(progress * 100) % 10 == 0:  # Actualizar cada 10%
                        try:
                            self.on_progress(progress)
                        except Exception:
                            pass

                self.frame_count += 1
                fps_counter += 1

                # Calcular FPS cada cierto intervalo
                if fps_counter >= fps_update_interval:
                    current_time = time.time()
                    if current_time > start_time:
                        fps = fps_counter / (current_time - start_time)
                        self.fps_samples.append(fps)
                        if len(self.fps_samples) > 10:
                            self.fps_samples.pop(0)

                    start_time = current_time
                    fps_counter = 0

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

            # Destruir solo la ventana activa si existe
            if self.active_window_name:
                try:
                    cv2.destroyWindow(self.active_window_name)
                    print(f"🚮 Ventana {self.active_window_name} cerrada")
                except Exception:
                    pass
                cv2.destroyAllWindows()
            else:
                cv2.destroyAllWindows()

            if self.csv_fp:
                try:
                    self.csv_fp.flush()
                except Exception:
                    pass
                self.csv_fp.close()

            # Aplicar características avanzadas de MLflow al finalizar
            if detector is not None:
                self._current_model = detector.model
            self._apply_advanced_mlflow_features()

            if self.on_finish:
                try:
                    self.on_finish()
                except Exception:
                    pass



