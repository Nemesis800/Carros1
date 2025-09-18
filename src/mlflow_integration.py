# src/mlflow_integration.py
"""
Integración de MLflow para el sistema de detección y conteo de vehículos.

- Usa la carpeta 'mlruns/' en el directorio raíz del proyecto (como 'reports/').
- Construye el tracking URI con Path(...).as_uri() para que sea portable (Windows/Linux).
- Provee un tracker con helpers para registrar parámetros, métricas y artefactos.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import json
import time

import mlflow
from ultralytics import YOLO


class VehicleDetectionMLflowTracker:
    """Tracker de MLflow especializado en el dominio de detección de vehículos."""

    def __init__(
        self,
        experiment_name: str = "Contador-Vehiculos",
        tracking_subdir: str = "mlruns",
    ):
        """Inicializa el tracker de MLflow usando 'mlruns/' en el raíz del proyecto."""
        self.experiment_name = experiment_name

        # Carpeta mlruns en el root del repo (igual filosofía que 'reports/')
        root = Path(__file__).resolve().parents[1]
        mlruns_dir = (root / tracking_subdir).resolve()
        mlruns_dir.mkdir(exist_ok=True)

        # URI portable (file:///C:/... en Windows, file:///... en Linux)
        mlflow.set_tracking_uri(mlruns_dir.as_uri())
        mlflow.set_experiment(experiment_name)

        # Estado interno
        self.run_id: Optional[str] = None
        self.start_time: Optional[float] = None
        self.total_detections = 0
        self.total_frames_processed = 0
        self.fps_samples: List[float] = []

        print(f"🔧 MLflow tracking: {mlruns_dir.as_uri()}  |  exp: {experiment_name}")

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------
    def start_experiment_run(self, config, video_source: str | int, tags: Dict[str, Any] | None = None) -> str:
        """Inicia un run (cierra el anterior si existía)."""
        if self.run_id is not None:
            self.end_experiment_run()

        default_tags = {
            "mlflow.source.type": "LOCAL",
            "mlflow.source.name": "vehicle_detection_system",
            "video_type": "webcam" if isinstance(video_source, int) else "file",
            "model_family": "YOLO",
            "task_type": "object_detection_counting",
            "timestamp": datetime.now().isoformat(),
            "framework": "ultralytics",
        }
        if tags:
            default_tags.update(tags)

        run = mlflow.start_run(tags=default_tags)
        self.run_id = run.info.run_id
        self.start_time = time.time()

        self._log_configuration_parameters(config, video_source)
        print(f"🚀 MLflow Run: {self.run_id}")
        return self.run_id

    def end_experiment_run(self, status: str = "FINISHED") -> None:
        """Finaliza el run actual y limpia estado interno."""
        if not self.run_id:
            return
        try:
            if self.start_time:
                total_time = time.time() - self.start_time
                mlflow.log_metric("total_experiment_duration_seconds", total_time)
            mlflow.end_run(status=status)
            print(f"🏁 Run finalizado ({status})")
        finally:
            self.run_id = None
            self.start_time = None
            self.total_detections = 0
            self.total_frames_processed = 0
            self.fps_samples = []

    # ------------------------------------------------------------------
    # Parámetros / Metadatos
    # ------------------------------------------------------------------
    def _log_configuration_parameters(self, config, video_source: str | int) -> None:
        """Registra parámetros de AppConfig."""
        params = {
            "model_name": getattr(config, "model_name", ""),
            "confidence_threshold": getattr(config, "conf", ""),
            "iou_threshold": getattr(config, "iou", ""),
            "device": getattr(config, "device", "auto") or "auto",
            "line_orientation": getattr(config, "line_orientation", ""),
            "line_position": getattr(config, "line_position", ""),
            "invert_direction": getattr(config, "invert_direction", ""),
            "capacity_car": getattr(config, "capacity_car", ""),
            "capacity_moto": getattr(config, "capacity_moto", ""),
            "enable_csv": getattr(config, "enable_csv", False),
            "csv_directory": getattr(config, "csv_dir", ""),
            "video_source_type": "webcam" if isinstance(video_source, int) else "file",
            "video_source": str(video_source),
        }
        mlflow.log_params(params)

    def log_model_metadata(self, model: YOLO) -> None:
        """Registra metadatos simples del modelo YOLO."""
        if not self.run_id:
            return
        try:
            names = getattr(model, "names", {})
            if isinstance(names, dict):
                class_names = list(names.values())
                num_classes = len(names)
            else:
                class_names = list(names)
                num_classes = len(class_names)

            mlflow.log_params(
                {
                    "model_type": "YOLO",
                    "num_classes": num_classes,
                    "model_size_hint": "nano",
                }
            )
            # Guardar clases como artefacto JSON
            data = {"classes": class_names, "num_classes": num_classes}
            tmp = Path("classes.json")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            mlflow.log_artifact(str(tmp), artifact_path="model_metadata")
            tmp.unlink(missing_ok=True)
        except Exception as e:
            mlflow.log_param("model_metadata_error", str(e))

    # ------------------------------------------------------------------
    # Métricas
    # ------------------------------------------------------------------
    def log_detection_metrics(
        self,
        frame_detections: int,
        car_count: int,
        moto_count: int,
        fps: float | None = None,
        processing_time: float | None = None,
    ) -> None:
        if not self.run_id:
            return
        self.total_detections += frame_detections
        self.total_frames_processed += 1
        if fps is not None:
            self.fps_samples.append(fps)

        metrics: Dict[str, float] = {
            "detections_per_frame": float(frame_detections),
            "cars_detected": float(car_count),
            "motorcycles_detected": float(moto_count),
        }
        if fps is not None:
            metrics["current_fps"] = float(fps)
        if processing_time is not None:
            metrics["frame_processing_time"] = float(processing_time)
        if self.total_frames_processed:
            metrics["avg_detections_per_frame"] = self.total_detections / self.total_frames_processed
        if self.fps_samples:
            metrics["avg_fps"] = sum(self.fps_samples) / len(self.fps_samples)

        step = int((time.time() - (self.start_time or time.time())) * 1000)
        mlflow.log_metrics(metrics, step=step)

    def log_counting_events(
        self,
        car_in: int,
        car_out: int,
        car_inventory: int,
        moto_in: int,
        moto_out: int,
        moto_inventory: int,
        capacity_exceeded: bool = False,
    ) -> None:
        if not self.run_id:
            return
        metrics = {
            "cars_entered_total": car_in,
            "cars_exited_total": car_out,
            "cars_current_inventory": car_inventory,
            "motorcycles_entered_total": moto_in,
            "motorcycles_exited_total": moto_out,
            "motorcycles_current_inventory": moto_inventory,
            "total_vehicles_inside": car_inventory + moto_inventory,
            "net_vehicle_flow": (car_in + moto_in) - (car_out + moto_out),
        }
        if capacity_exceeded:
            metrics["capacity_exceeded"] = 1
        step = int((time.time() - (self.start_time or time.time())) * 1000)
        mlflow.log_metrics(metrics, step=step)

    def log_system_performance(
        self,
        total_processing_time: float,
        total_frames: int,
        memory_usage_mb: float | None = None,
    ) -> None:
        if not self.run_id:
            return
        metrics = {
            "total_processing_time_seconds": float(total_processing_time),
            "total_frames_processed": int(total_frames),
            "average_fps": (total_frames / max(total_processing_time, 1.0)),
        }
        if memory_usage_mb is not None:
            metrics["peak_memory_usage_mb"] = float(memory_usage_mb)
        mlflow.log_metrics(metrics)


# ----------------------------------------------------------------------
# Helpers globales
# ----------------------------------------------------------------------
_global_tracker: Optional[VehicleDetectionMLflowTracker] = None


def get_mlflow_tracker() -> VehicleDetectionMLflowTracker:
    """Obtiene (o crea) la instancia global del tracker."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = VehicleDetectionMLflowTracker()
    return _global_tracker


def initialize_mlflow_tracking(
    experiment_name: str = "Contador-Vehiculos",
    tracking_subdir: str = "mlruns",
) -> VehicleDetectionMLflowTracker:
    """Reinicia/crea el tracker global con experimento y carpeta dados."""
    global _global_tracker
    _global_tracker = VehicleDetectionMLflowTracker(experiment_name, tracking_subdir)
    return _global_tracker

