"""
Integración de MLflow para el sistema de detección y conteo de vehículos.
Maneja el seguimiento de experimentos, métricas, parámetros y artefactos.
"""

from __future__ import annotations
import os
import json
import time
import shutil
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

import mlflow
import mlflow.pyfunc
import numpy as np
from ultralytics import YOLO

from config import AppConfig


class VehicleDetectionMLflowTracker:
    """
    Tracker de MLflow especializado para el sistema de detección de vehículos.
    Maneja experimentos, métricas, parámetros y artefactos específicos del dominio.
    """
    
    def __init__(self, experiment_name: str = "vehicle_detection_system", tracking_uri: str = None):
        """
        Inicializa el tracker de MLflow.
        
        Args:
            experiment_name: Nombre del experimento en MLflow
            tracking_uri: URI del servidor de tracking (None = local)
        """
        self.experiment_name = experiment_name
        
        # Configurar MLflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            # Por defecto usar directorio local
            mlflow_dir = Path("mlruns")
            mlflow_dir.mkdir(exist_ok=True)
            mlflow.set_tracking_uri(f"file:///{mlflow_dir.absolute()}")
        
        print(f"🔧 MLflow configurado en: {mlflow.get_tracking_uri()}")
        
        # Crear o obtener experimento
        try:
            self.experiment_id = mlflow.create_experiment(experiment_name)
            print(f"✅ Experimento '{experiment_name}' creado con ID: {self.experiment_id}")
        except mlflow.exceptions.MlflowException:
            # El experimento ya existe
            experiment = mlflow.get_experiment_by_name(experiment_name)
            self.experiment_id = experiment.experiment_id
            print(f"📂 Usando experimento existente '{experiment_name}' con ID: {self.experiment_id}")
        
        mlflow.set_experiment(experiment_name)
        
        self.run_id: Optional[str] = None
        self.start_time: Optional[float] = None
        
        # Contadores para métricas acumuladas
        self.total_detections = 0
        self.total_frames_processed = 0
        self.fps_samples = []
        
    def start_experiment_run(self, 
                           config: AppConfig, 
                           video_source: str | int, 
                           tags: Dict[str, Any] = None) -> str:
        """
        Inicia un nuevo run de MLflow con la configuración del experimento.
        
        Args:
            config: Configuración de la aplicación
            video_source: Fuente de video (webcam o path)
            tags: Tags adicionales para el run
            
        Returns:
            run_id: ID del run creado
        """
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
        
        # Iniciar run
        run = mlflow.start_run(
            experiment_id=self.experiment_id,
            tags=default_tags
        )
        
        self.run_id = run.info.run_id
        self.start_time = time.time()
        
        print(f"🚀 MLflow Run iniciado: {self.run_id}")
        
        # Registrar parámetros de configuración
        self._log_configuration_parameters(config, video_source)
        
        return self.run_id
    
    def _log_configuration_parameters(self, config: AppConfig, video_source: str | int):
        """Registra los parámetros de configuración en MLflow."""
        params = {
            # Parámetros del modelo
            "model_name": config.model_name,
            "confidence_threshold": config.conf,
            "iou_threshold": config.iou,
            "device": config.device or "auto",
            
            # Parámetros de la línea de conteo
            "line_orientation": config.line_orientation,
            "line_position": config.line_position,
            "invert_direction": config.invert_direction,
            
            # Capacidades del sistema
            "capacity_car": config.capacity_car,
            "capacity_moto": config.capacity_moto,
            
            # Configuración de salida
            "enable_csv": config.enable_csv,
            "csv_directory": config.csv_dir,
            
            # Fuente de video
            "video_source_type": "webcam" if isinstance(video_source, int) else "file",
            "video_source": str(video_source),
        }
        
        mlflow.log_params(params)
        print(f"📝 Parámetros de configuración registrados: {len(params)} parámetros")
    
    def log_model_metadata(self, model: YOLO):
        """
        Registra metadatos del modelo YOLO en MLflow.
        
        Args:
            model: Instancia del modelo YOLO
        """
        if not self.run_id:
            print("⚠️  Advertencia: No hay run activo para registrar el modelo")
            return
        
        try:
            # Información del modelo
            model_info = {
                "model_type": "YOLO",
                "model_architecture": getattr(model, 'model_name', 'unknown'),
                "input_size": 640,  # YOLO standard
                "num_classes": len(model.names) if hasattr(model, 'names') else 0,
                "class_names": list(model.names.values()) if hasattr(model, 'names') else [],
                "model_size": "nano",  # Inferido del nombre
            }
            
            # Registrar como parámetros
            mlflow.log_params({
                f"model_{k}": str(v) if not isinstance(v, (int, float, bool)) else v 
                for k, v in model_info.items() if k != "class_names"
            })
            
            # Registrar clases como artefacto JSON
            classes_info = {
                "classes": model_info.get("class_names", []),
                "num_classes": model_info.get("num_classes", 0)
            }
            
            classes_file = Path("temp_classes.json")
            with open(classes_file, "w") as f:
                json.dump(classes_info, f, indent=2)
            
            mlflow.log_artifact(str(classes_file), "model_metadata")
            classes_file.unlink()  # Eliminar archivo temporal
            
            # Intentar registrar el archivo del modelo como artefacto
            model_file = f"{model.model_name}" if hasattr(model, 'model_name') else None
            if model_file and os.path.exists(model_file):
                mlflow.log_artifact(model_file, "model_weights")
                print(f"📦 Modelo registrado como artefacto: {model_file}")
            
            print("🤖 Metadatos del modelo registrados exitosamente")
                
        except Exception as e:
            mlflow.log_param("model_registration_error", str(e))
            print(f"❌ Error registrando metadatos del modelo: {e}")
    
    def log_detection_metrics(self, 
                            frame_detections: int, 
                            car_count: int, 
                            moto_count: int,
                            fps: float = None,
                            processing_time: float = None):
        """
        Registra métricas de detección en tiempo real.
        
        Args:
            frame_detections: Número de detecciones en este frame
            car_count: Número de carros detectados
            moto_count: Número de motos detectadas  
            fps: Frames por segundo
            processing_time: Tiempo de procesamiento del frame
        """
        if not self.run_id:
            return
        
        try:
            self.total_detections += frame_detections
            self.total_frames_processed += 1
            
            if fps is not None:
                self.fps_samples.append(fps)
            
            # Usar timestamp relativo desde el inicio
            timestamp = int((time.time() - self.start_time) * 1000) if self.start_time else None
            
            # Métricas por frame
            metrics = {
                "detections_per_frame": frame_detections,
                "cars_detected": car_count,
                "motorcycles_detected": moto_count,
            }
            
            if fps is not None:
                metrics["current_fps"] = fps
                
            if processing_time is not None:
                metrics["frame_processing_time"] = processing_time
            
            # Métricas acumuladas
            if self.total_frames_processed > 0:
                metrics["avg_detections_per_frame"] = self.total_detections / self.total_frames_processed
                
            if self.fps_samples:
                metrics["avg_fps"] = sum(self.fps_samples) / len(self.fps_samples)
            
            # Registrar todas las métricas con step
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=timestamp)
                
        except Exception as e:
            print(f"❌ Error registrando métricas de detección: {e}")
    
    def log_counting_events(self, 
                          car_in: int, car_out: int, car_inventory: int,
                          moto_in: int, moto_out: int, moto_inventory: int,
                          capacity_exceeded: bool = False):
        """
        Registra métricas de conteo de vehículos.
        
        Args:
            car_in, car_out, car_inventory: Conteos de carros
            moto_in, moto_out, moto_inventory: Conteos de motos
            capacity_exceeded: Si se excedió la capacidad
        """
        if not self.run_id:
            return
        
        try:
            timestamp = int((time.time() - self.start_time) * 1000) if self.start_time else None
            
            # Métricas de conteo
            counting_metrics = {
                # Conteos absolutos
                "cars_entered_total": car_in,
                "cars_exited_total": car_out, 
                "cars_current_inventory": car_inventory,
                "motorcycles_entered_total": moto_in,
                "motorcycles_exited_total": moto_out,
                "motorcycles_current_inventory": moto_inventory,
                
                # Métricas derivadas
                "total_vehicles_inside": car_inventory + moto_inventory,
                "total_entries": car_in + moto_in,
                "total_exits": car_out + moto_out,
                "net_vehicle_flow": (car_in + moto_in) - (car_out + moto_out),
                
                # Ratios
                "car_to_moto_ratio": car_inventory / max(moto_inventory, 1),
                "occupancy_rate": (car_inventory + moto_inventory) / max(car_in + moto_in, 1),
            }
            
            if capacity_exceeded:
                counting_metrics["capacity_exceeded"] = 1
            
            # Registrar métricas
            for key, value in counting_metrics.items():
                mlflow.log_metric(key, value, step=timestamp)
                
        except Exception as e:
            print(f"❌ Error registrando métricas de conteo: {e}")
    
    def log_system_performance(self, 
                             total_processing_time: float,
                             total_frames: int,
                             memory_usage_mb: float = None):
        """
        Registra métricas de rendimiento del sistema.
        
        Args:
            total_processing_time: Tiempo total de procesamiento
            total_frames: Número total de frames procesados
            memory_usage_mb: Uso de memoria en MB
        """
        if not self.run_id:
            return
        
        try:
            performance_metrics = {
                "total_processing_time_seconds": total_processing_time,
                "total_frames_processed": total_frames,
                "average_fps": total_frames / total_processing_time if total_processing_time > 0 else 0,
                "processing_efficiency": total_frames / max(total_processing_time, 1),
            }
            
            if memory_usage_mb is not None:
                performance_metrics["peak_memory_usage_mb"] = memory_usage_mb
            
            mlflow.log_metrics(performance_metrics)
            print(f"⚡ Métricas de rendimiento registradas: {len(performance_metrics)} métricas")
            
        except Exception as e:
            print(f"❌ Error registrando métricas de rendimiento: {e}")
    
    def log_output_artifacts(self, csv_path: str = None, video_sample_path: str = None):
        """
        Registra artefactos de salida (CSV, muestras de video, etc.).
        
        Args:
            csv_path: Ruta al archivo CSV de resultados
            video_sample_path: Ruta a muestra de video procesado
        """
        if not self.run_id:
            return
        
        try:
            artifacts_logged = []
            
            # Registrar CSV de resultados
            if csv_path and os.path.exists(csv_path):
                mlflow.log_artifact(csv_path, "reports")
                artifacts_logged.append("CSV report")
            
            # Registrar muestra de video (si no es muy grande)
            if video_sample_path and os.path.exists(video_sample_path):
                file_size_mb = os.path.getsize(video_sample_path) / (1024 * 1024)
                
                if file_size_mb <= 100:  # Solo si es menor a 100MB
                    mlflow.log_artifact(video_sample_path, "video_samples")
                    artifacts_logged.append(f"Video sample ({file_size_mb:.1f}MB)")
                else:
                    mlflow.log_param("video_sample_size_mb", file_size_mb)
                    mlflow.log_param("video_sample_path", video_sample_path)
                    artifacts_logged.append(f"Video metadata (too large: {file_size_mb:.1f}MB)")
            
            if artifacts_logged:
                print(f"📎 Artefactos registrados: {', '.join(artifacts_logged)}")
                
        except Exception as e:
            print(f"❌ Error registrando artefactos: {e}")
    
    def register_model_in_registry(self, model_name: str = "vehicle_detection_yolo", stage: str = "None"):
        """
        Registra el modelo en el MLflow Model Registry.
        
        Args:
            model_name: Nombre del modelo en el registry
            stage: Etapa del modelo (None, Staging, Production, Archived)
        """
        if not self.run_id:
            print("⚠️  No hay run activo para registrar el modelo")
            return
        
        try:
            model_uri = f"runs:/{self.run_id}/model_weights"
            
            # Registrar modelo
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
                tags={
                    "framework": "ultralytics", 
                    "task": "object_detection",
                    "domain": "vehicle_counting",
                    "registration_date": datetime.now().isoformat()
                }
            )
            
            print(f"🏷️  Modelo registrado: {model_name} v{model_version.version}")
            
            # Transicionar a la etapa especificada si no es None
            if stage.lower() != "none":
                from mlflow.tracking import MlflowClient
                client = MlflowClient()
                client.transition_model_version_stage(
                    name=model_name,
                    version=model_version.version,
                    stage=stage
                )
                print(f"🔄 Modelo transicionado a etapa: {stage}")
                
        except Exception as e:
            print(f"❌ Error registrando modelo en registry: {e}")
    
    def end_experiment_run(self, status: str = "FINISHED"):
        """
        Finaliza el run actual de MLflow.
        
        Args:
            status: Estado final del run (FINISHED, FAILED, KILLED)
        """
        if self.run_id:
            try:
                # Registrar tiempo total si tenemos start_time
                if self.start_time:
                    total_time = time.time() - self.start_time
                    mlflow.log_metric("total_experiment_duration_seconds", total_time)
                    
                    # Resumen final
                    summary_metrics = {
                        "final_total_detections": self.total_detections,
                        "final_frames_processed": self.total_frames_processed,
                        "final_avg_fps": sum(self.fps_samples) / len(self.fps_samples) if self.fps_samples else 0,
                    }
                    mlflow.log_metrics(summary_metrics)
                
                mlflow.end_run(status=status)
                
                print(f"🏁 MLflow Run finalizado con estado: {status}")
                print(f"🔗 URL del run: {self.get_run_url()}")
                
                self.run_id = None
                self.start_time = None
                
                # Reset contadores
                self.total_detections = 0
                self.total_frames_processed = 0
                self.fps_samples = []
                
            except Exception as e:
                print(f"❌ Error finalizando MLflow run: {e}")

    def get_experiment_url(self) -> str:
        """Retorna la URL del experimento en MLflow."""
        tracking_uri = mlflow.get_tracking_uri()
        if tracking_uri.startswith("file:///"):
            # Para URI local, usar localhost
            return f"http://localhost:5000/#/experiments/{self.experiment_id}"
        else:
            return f"{tracking_uri}/#/experiments/{self.experiment_id}"
    
    def get_run_url(self) -> str:
        """Retorna la URL del run actual en MLflow."""
        if self.run_id:
            tracking_uri = mlflow.get_tracking_uri()
            if tracking_uri.startswith("file:///"):
                return f"http://localhost:5000/#/experiments/{self.experiment_id}/runs/{self.run_id}"
            else:
                return f"{tracking_uri}/#/experiments/{self.experiment_id}/runs/{self.run_id}"
        return ""

    def launch_mlflow_ui(self, port: int = 5000):
        """
        Lanza la interfaz web de MLflow.
        
        Args:
            port: Puerto para la interfaz web
        """
        import subprocess
        import sys
        
        try:
            print(f"🌐 Lanzando MLflow UI en puerto {port}...")
            print(f"📂 Experimentos disponibles en: http://localhost:{port}")
            
            # Ejecutar MLflow UI
            subprocess.Popen([
                sys.executable, "-m", "mlflow", "ui", 
                "--port", str(port),
                "--backend-store-uri", mlflow.get_tracking_uri()
            ])
            
        except Exception as e:
            print(f"❌ Error lanzando MLflow UI: {e}")


# Instancia global para fácil acceso
_global_tracker: Optional[VehicleDetectionMLflowTracker] = None

def get_mlflow_tracker() -> VehicleDetectionMLflowTracker:
    """Obtiene o crea la instancia global del tracker de MLflow."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = VehicleDetectionMLflowTracker()
    return _global_tracker

def initialize_mlflow_tracking(experiment_name: str = "vehicle_detection_system", 
                             tracking_uri: str = None) -> VehicleDetectionMLflowTracker:
    """
    Inicializa el tracker global de MLflow.
    
    Args:
        experiment_name: Nombre del experimento
        tracking_uri: URI del servidor de tracking
        
    Returns:
        VehicleDetectionMLflowTracker: Instancia del tracker
    """
    global _global_tracker
    _global_tracker = VehicleDetectionMLflowTracker(experiment_name, tracking_uri)
    return _global_tracker