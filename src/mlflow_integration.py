# src/mlflow_integration.py
"""
Integración de MLflow para el sistema de detección y conteo de vehículos.

Provee un tracker especializado (`VehicleDetectionMLflowTracker`) que maneja:
- Seguimiento de experimentos.
- Registro de parámetros, métricas y artefactos.
- Registro de modelos en el Model Registry de MLflow.
- Visualizaciones automáticas de rendimiento.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
import mlflow
import mlflow.pyfunc
import mlflow.pytorch
import numpy as np
import psutil
import seaborn as sns
import torch
from ultralytics import YOLO

from config import AppConfig


class VehicleDetectionMLflowTracker:
    """Tracker de MLflow especializado en el dominio de detección de vehículos."""

    def __init__(self, experiment_name: str = "vehicle_detection_system", tracking_uri: str | None = None):
        """Inicializa el tracker de MLflow.

        Args:
            experiment_name: Nombre del experimento.
            tracking_uri: URI del servidor de tracking (None = local).
        """
        self.experiment_name = experiment_name

        # Configuración del tracking
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            mlflow_dir = Path("mlruns")
            mlflow_dir.mkdir(exist_ok=True)
            mlflow.set_tracking_uri(f"file:///{mlflow_dir.absolute()}")

        print(f"🔧 MLflow configurado en: {mlflow.get_tracking_uri()}")

        # Crear o recuperar experimento
        try:
            self.experiment_id = mlflow.create_experiment(experiment_name)
            print(f"✅ Experimento '{experiment_name}' creado con ID: {self.experiment_id}")
        except mlflow.exceptions.MlflowException:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            self.experiment_id = experiment.experiment_id
            print(f"📂 Usando experimento existente '{experiment_name}' con ID: {self.experiment_id}")

        mlflow.set_experiment(experiment_name)

        # Estado interno
        self.run_id: Optional[str] = None
        self.start_time: Optional[float] = None
        self.total_detections = 0
        self.total_frames_processed = 0
        self.fps_samples: List[float] = []

    # ------------------------------------------------------------------
    # Inicialización y configuración
    # ------------------------------------------------------------------
    def start_experiment_run(self, config: AppConfig, video_source: str | int, tags: Dict[str, Any] | None = None) -> str:
        """Inicia un nuevo run de MLflow con la configuración del experimento."""
        if self.run_id is not None:
            self.end_experiment_run()

        # Tags por defecto
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

        run = mlflow.start_run(experiment_id=self.experiment_id, tags=default_tags)
        self.run_id = run.info.run_id
        self.start_time = time.time()

        print(f"🚀 MLflow Run iniciado: {self.run_id}")
        self._log_configuration_parameters(config, video_source)
        return self.run_id

    def _log_configuration_parameters(self, config: AppConfig, video_source: str | int) -> None:
        """Registra los parámetros de configuración de AppConfig en MLflow."""
        params = {
            "model_name": config.model_name,
            "confidence_threshold": config.conf,
            "iou_threshold": config.iou,
            "device": config.device or "auto",
            "line_orientation": config.line_orientation,
            "line_position": config.line_position,
            "invert_direction": config.invert_direction,
            "capacity_car": config.capacity_car,
            "capacity_moto": config.capacity_moto,
            "enable_csv": config.enable_csv,
            "csv_directory": config.csv_dir,
            "video_source_type": "webcam" if isinstance(video_source, int) else "file",
            "video_source": str(video_source),
        }
        mlflow.log_params(params)
        print(f"📝 Parámetros registrados: {len(params)}")

    # ------------------------------------------------------------------
    # Registro de modelo y metadatos
    # ------------------------------------------------------------------
    def log_model_metadata(self, model: YOLO) -> None:
        """Registra metadatos y clases de un modelo YOLO en MLflow."""
        if not self.run_id:
            print("⚠️ No hay run activo para registrar el modelo")
            return
        try:
            model_info = {
                "model_type": "YOLO",
                "model_architecture": getattr(model, "model_name", "unknown"),
                "input_size": 640,
                "num_classes": len(model.names) if hasattr(model, "names") else 0,
                "class_names": list(model.names.values()) if hasattr(model, "names") else [],
                "model_size": "nano",
            }
            mlflow.log_params({f"model_{k}": v for k, v in model_info.items() if k != "class_names"})

            # Guardar clases como artefacto
            classes_info = {"classes": model_info.get("class_names", []), "num_classes": model_info.get("num_classes", 0)}
            classes_file = Path("temp_classes.json")
            with open(classes_file, "w") as f:
                json.dump(classes_info, f, indent=2)
            mlflow.log_artifact(str(classes_file), "model_metadata")
            classes_file.unlink()

            print("🤖 Metadatos del modelo registrados")
        except Exception as e:
            mlflow.log_param("model_registration_error", str(e))
            print(f"❌ Error registrando metadatos: {e}")

    # ------------------------------------------------------------------
    # Logging de métricas
    # ------------------------------------------------------------------
    def log_detection_metrics(self, frame_detections: int, car_count: int, moto_count: int, fps: float | None = None, processing_time: float | None = None) -> None:
        """Registra métricas de detección por frame y acumuladas."""
        if not self.run_id:
            return
        try:
            self.total_detections += frame_detections
            self.total_frames_processed += 1
            if fps is not None:
                self.fps_samples.append(fps)

            metrics = {
                "detections_per_frame": frame_detections,
                "cars_detected": car_count,
                "motorcycles_detected": moto_count,
            }
            if fps is not None:
                metrics["current_fps"] = fps
            if processing_time is not None:
                metrics["frame_processing_time"] = processing_time
            if self.total_frames_processed > 0:
                metrics["avg_detections_per_frame"] = self.total_detections / self.total_frames_processed
            if self.fps_samples:
                metrics["avg_fps"] = sum(self.fps_samples) / len(self.fps_samples)

            mlflow.log_metrics(metrics, step=int((time.time() - self.start_time) * 1000))
        except Exception as e:
            print(f"❌ Error registrando métricas: {e}")

    def log_counting_events(self, car_in: int, car_out: int, car_inventory: int, moto_in: int, moto_out: int, moto_inventory: int, capacity_exceeded: bool = False) -> None:
        """Registra métricas de conteo de vehículos."""
        if not self.run_id:
            return
        try:
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
            mlflow.log_metrics(metrics, step=int((time.time() - self.start_time) * 1000))
        except Exception as e:
            print(f"❌ Error registrando conteo: {e}")

    def log_system_performance(self, total_processing_time: float, total_frames: int, memory_usage_mb: float | None = None) -> None:
        """Registra métricas globales de rendimiento."""
        if not self.run_id:
            return
        try:
            metrics = {
                "total_processing_time_seconds": total_processing_time,
                "total_frames_processed": total_frames,
                "average_fps": total_frames / max(total_processing_time, 1),
            }
            if memory_usage_mb is not None:
                metrics["peak_memory_usage_mb"] = memory_usage_mb
            mlflow.log_metrics(metrics)
            print("⚡ Métricas de rendimiento registradas")
        except Exception as e:
            print(f"❌ Error registrando rendimiento: {e}")

    # ------------------------------------------------------------------
    # Finalización y utilidades
    # ------------------------------------------------------------------
    def end_experiment_run(self, status: str = "FINISHED") -> None:
        """Finaliza el run actual de MLflow y resetea contadores."""
        if not self.run_id:
            return
        try:
            if self.start_time:
                total_time = time.time() - self.start_time
                mlflow.log_metric("total_experiment_duration_seconds", total_time)
            mlflow.end_run(status=status)
            print(f"🏁 Run finalizado ({status})")
            self.run_id = None
            self.start_time = None
            self.total_detections = 0
            self.total_frames_processed = 0
            self.fps_samples = []
        except Exception as e:
            print(f"❌ Error finalizando run: {e}")


# ----------------------------------------------------------------------
# Instancia global para acceso fácil
# ----------------------------------------------------------------------
_global_tracker: Optional[VehicleDetectionMLflowTracker] = None


def get_mlflow_tracker() -> VehicleDetectionMLflowTracker:
    """Obtiene (o crea) la instancia global del tracker de MLflow."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = VehicleDetectionMLflowTracker()
    return _global_tracker


def initialize_mlflow_tracking(experiment_name: str = "vehicle_detection_system", tracking_uri: str | None = None) -> VehicleDetectionMLflowTracker:
    """Inicializa el tracker global de MLflow con un experimento específico."""
    global _global_tracker
    _global_tracker = VehicleDetectionMLflowTracker(experiment_name, tracking_uri)
    return _global_tracker

