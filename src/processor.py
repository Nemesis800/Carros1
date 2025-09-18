from __future__ import annotations
import os
import csv
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

import cv2
import numpy as np
import supervision as sv

# Importar MLflow con manejo de errores
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("⚠️  MLflow no está instalado. Las métricas no se registrarán.")

from detector import VehicleDetector
from counter import LineCrossingCounterByClass
from config import AppConfig, WINDOW_NAME, sanitize_filename
from utils import winsound_beep


class VideoProcessor(threading.Thread):
    """
    Procesa un stream de video aplicando detección, tracking y conteo.
    Si display=False, corre en modo headless (sin ventanas), ideal para CLI o Streamlit.

    Nuevos callbacks opcionales:
      - on_frame(frame_rgb: np.ndarray): frame anotado en RGB para UI web.
      - on_progress(p: float): progreso [0..1] según frames procesados (si se conoce el total).
    """
    def __init__(
        self,
        video_source: int | str,              # 0 (webcam) o ruta a video
        config: AppConfig,                    # parámetros de detección/contador/CSV
        stop_event: threading.Event,          # bandera para interrupción segura
        on_error: Callable[[str], None] | None = None,     # callback de error (UI/CLI)
        on_finish: Callable[[], None] | None = None,       # callback al terminar
        display: bool = True,                 # True=con ventanas; False=headless
        on_frame: Callable[[np.ndarray], None] | None = None,     # NUEVO
        on_progress: Callable[[float], None] | None = None,       # NUEVO
        # MLflow parameters
        enable_mlflow: bool = True,           # Habilitar tracking con MLflow
        experiment_name: str = "vehicle_detection",  # Nombre del experimento MLflow
        mlflow_tags: dict = None,            # Tags adicionales para MLflow
    ) -> None:
        super().__init__(daemon=True)
        self.video_source = video_source
        self.config = config
        self.stop_event = stop_event
        self.on_error = on_error
        self.on_finish = on_finish
        self.display = display

        # NUEVO: hooks para UI web
        self.on_frame = on_frame
        self.on_progress = on_progress

        # MLflow configuration
        self.enable_mlflow = enable_mlflow and MLFLOW_AVAILABLE
        self.experiment_name = experiment_name
        self.mlflow_tags = mlflow_tags or {}
        self.mlflow_run_id = None
        self.mlflow_start_time = None
        self.frame_count = 0
        self.detection_count = 0
        self.fps_samples = []
        
        # Estados para alarma de capacidad (evitar beeps repetidos por frame)
        self._prev_over_car = False
        self._prev_over_moto = False

        # CSV y acumuladores de conteo (para detectar incrementos)
        self.csv_fp = None
        self.csv_writer = None
        self._prev_counts = {"car_in": 0, "car_out": 0, "moto_in": 0, "moto_out": 0}
        self._last_car_inv = 0
        self._last_moto_inv = 0
        self._csv_path_str = None  # ruta del CSV generado (informativa)

    # ---------- Utilidades internas ----------
    def _notify_error(self, msg: str) -> None:
        """Envía error al callback si existe; si no, imprime en consola."""
        # Log error to MLflow if available
        if self.enable_mlflow and self.mlflow_run_id:
            try:
                mlflow.log_param("last_error", msg)
            except Exception:
                pass
        
        try:
            if self.on_error:
                self.on_error(msg)
            else:
                print(f"[ERROR] {msg}")
        except Exception:
            print(f"[ERROR] {msg}")
    
    # ---------- Métodos MLflow ----------
    def _init_mlflow(self) -> None:
        """Inicializa MLflow experiment y run."""
        if not self.enable_mlflow:
            return
        
        try:
            # Configurar directorio MLflow local
            mlflow_dir = Path("mlruns")
            mlflow_dir.mkdir(exist_ok=True)
            mlflow.set_tracking_uri(f"file:///{mlflow_dir.absolute()}")
            
            # Crear o usar experimento existente
            try:
                experiment_id = mlflow.create_experiment(self.experiment_name)
                print(f"🎦 Experimento MLflow creado: {self.experiment_name}")
            except mlflow.exceptions.MlflowException:
                experiment = mlflow.get_experiment_by_name(self.experiment_name)
                experiment_id = experiment.experiment_id
                print(f"📂 Usando experimento MLflow: {self.experiment_name}")
            
            mlflow.set_experiment(self.experiment_name)
            
            # Tags por defecto
            default_tags = {
                "mlflow.source.type": "LOCAL",
                "mlflow.source.name": "vehicle_detection_processor",
                "video_type": "webcam" if isinstance(self.video_source, int) else "file",
                "timestamp": datetime.now().isoformat(),
                "model_family": "YOLO",
                "task": "object_detection_counting"
            }
            default_tags.update(self.mlflow_tags)
            
            # Iniciar run
            run = mlflow.start_run(tags=default_tags)
            self.mlflow_run_id = run.info.run_id
            self.mlflow_start_time = time.time()
            
            # Log configuration parameters
            self._log_config_params()
            
            print(f"🚀 MLflow Run iniciado: {self.mlflow_run_id}")
            
        except Exception as e:
            print(f"⚠️  Error inicializando MLflow: {e}")
            self.enable_mlflow = False
    
    def _log_config_params(self) -> None:
        """Registra parámetros de configuración en MLflow."""
        if not self.enable_mlflow or not self.mlflow_run_id:
            return
        
        try:
            params = {
                "model_name": self.config.model_name,
                "confidence_threshold": self.config.conf,
                "iou_threshold": self.config.iou,
                "line_orientation": self.config.line_orientation,
                "line_position": self.config.line_position,
                "invert_direction": self.config.invert_direction,
                "capacity_car": self.config.capacity_car,
                "capacity_moto": self.config.capacity_moto,
                "enable_csv": getattr(self.config, 'enable_csv', False),
                "device": self.config.device or "auto",
                "video_source": str(self.video_source),
                "display_mode": self.display
            }
            
            mlflow.log_params(params)
            print(f"📝 Parámetros registrados en MLflow: {len(params)}")
            
        except Exception as e:
            print(f"⚠️  Error registrando parámetros: {e}")
    
    def _log_model_info(self, detector) -> None:
        """Registra información del modelo YOLO en MLflow."""
        if not self.enable_mlflow or not self.mlflow_run_id:
            return
        
        try:
            model = detector.model
            model_info = {
                "model_type": "YOLO",
                "model_architecture": self.config.model_name,
                "input_size": 640,
                "num_classes": len(model.names) if hasattr(model, 'names') else 0,
                "target_classes": list(detector.name_to_id.keys()) if hasattr(detector, 'name_to_id') else []
            }
            
            mlflow.log_params({
                f"model_{k}": str(v) if isinstance(v, list) else v
                for k, v in model_info.items() if k != "target_classes"
            })
            
            # Registrar clases como texto
            if model_info["target_classes"]:
                mlflow.log_text(",".join(model_info["target_classes"]), "target_classes.txt")
            
            # Registrar archivo del modelo si existe
            if os.path.exists(self.config.model_name):
                mlflow.log_artifact(self.config.model_name, "model")
            
            print(f"🤖 Modelo registrado en MLflow: {self.config.model_name}")
            
        except Exception as e:
            print(f"⚠️  Error registrando modelo: {e}")
    
    def _log_detection_metrics(self, detections_count: int, car_count: int, moto_count: int, fps: float = None) -> None:
        """Registra métricas de detección en MLflow."""
        if not self.enable_mlflow or not self.mlflow_run_id:
            return
        
        try:
            self.frame_count += 1
            self.detection_count += detections_count
            
            if fps is not None:
                self.fps_samples.append(fps)
            
            # Timestamp relativo desde inicio
            timestamp = int((time.time() - self.mlflow_start_time) * 1000) if self.mlflow_start_time else None
            
            # Métricas por frame (cada 30 frames para no saturar)
            if self.frame_count % 30 == 0:
                metrics = {
                    "detections_per_frame": detections_count,
                    "cars_detected": car_count,
                    "motorcycles_detected": moto_count,
                    "total_detections": self.detection_count,
                    "avg_detections_per_frame": self.detection_count / self.frame_count
                }
                
                if fps is not None:
                    metrics["current_fps"] = fps
                
                if self.fps_samples:
                    metrics["avg_fps"] = sum(self.fps_samples) / len(self.fps_samples)
                
                for key, value in metrics.items():
                    mlflow.log_metric(key, value, step=timestamp)
                    
        except Exception as e:
            print(f"⚠️  Error registrando métricas de detección: {e}")
    
    def _log_counting_metrics(self, car_in: int, car_out: int, car_inv: int, 
                             moto_in: int, moto_out: int, moto_inv: int) -> None:
        """Registra métricas de conteo en MLflow."""
        if not self.enable_mlflow or not self.mlflow_run_id:
            return
        
        try:
            timestamp = int((time.time() - self.mlflow_start_time) * 1000) if self.mlflow_start_time else None
            
            # Métricas de conteo (cada 60 frames)
            if self.frame_count % 60 == 0:
                metrics = {
                    "cars_in_total": car_in,
                    "cars_out_total": car_out,
                    "cars_inventory": car_inv,
                    "motorcycles_in_total": moto_in,
                    "motorcycles_out_total": moto_out,
                    "motorcycles_inventory": moto_inv,
                    "total_vehicles_inside": car_inv + moto_inv,
                    "total_entries": car_in + moto_in,
                    "total_exits": car_out + moto_out,
                    "net_flow": (car_in + moto_in) - (car_out + moto_out)
                }
                
                # Verificar exceso de capacidad
                if car_inv > self.config.capacity_car or moto_inv > self.config.capacity_moto:
                    metrics["capacity_exceeded"] = 1
                
                for key, value in metrics.items():
                    mlflow.log_metric(key, value, step=timestamp)
                    
        except Exception as e:
            print(f"⚠️  Error registrando métricas de conteo: {e}")
    
    def _finish_mlflow(self) -> None:
        """Finaliza el run de MLflow registrando métricas finales."""
        if not self.enable_mlflow or not self.mlflow_run_id:
            return
        
        try:
            # Métricas finales
            if self.mlflow_start_time:
                total_time = time.time() - self.mlflow_start_time
                final_metrics = {
                    "total_processing_time": total_time,
                    "total_frames_processed": self.frame_count,
                    "total_detections_final": self.detection_count,
                    "final_avg_fps": sum(self.fps_samples) / len(self.fps_samples) if self.fps_samples else 0,
                    "processing_efficiency": self.frame_count / total_time if total_time > 0 else 0
                }
                
                mlflow.log_metrics(final_metrics)
            
            # Registrar CSV como artefacto si existe
            if self._csv_path_str and os.path.exists(self._csv_path_str):
                mlflow.log_artifact(self._csv_path_str, "reports")
                print(f"📄 CSV registrado como artefacto: {self._csv_path_str}")
            
            # Finalizar run
            mlflow.end_run()
            print(f"🏁 MLflow Run finalizado exitosamente")
            print(f"🔗 Ver resultados en: http://localhost:5000/#/experiments")
            
        except Exception as e:
            print(f"⚠️  Error finalizando MLflow: {e}")

    def _init_csv(self) -> None:
        """Inicializa CSV si está habilitado en config (crea carpeta y cabecera)."""
        if not getattr(self.config, "enable_csv", False):
            return
        try:
            Path(self.config.csv_dir).mkdir(parents=True, exist_ok=True)
            ts_name = datetime.now().strftime("%Y%m%d_%H%M%S")
            src_name = "webcam" if isinstance(self.video_source, int) else Path(str(self.video_source)).name

            # Respeta nombre custom si se dio; de lo contrario usa timestamp+fuente
            if getattr(self.config, "csv_name", None):
                base = sanitize_filename(str(self.config.csv_name))
                filename = base if base.lower().endswith(".csv") else f"{base}.csv"
            else:
                filename = f"reporte_{ts_name}_{src_name}.csv"

            csv_path = Path(self.config.csv_dir) / filename

            # Evitar sobrescritura accidental: agrega sufijo incremental si existe
            if csv_path.exists():
                stem = csv_path.stem
                suffix = 1
                while True:
                    candidate = csv_path.with_name(f"{stem}_{suffix}.csv")
                    if not candidate.exists():
                        csv_path = candidate
                        break
                    suffix += 1

            # Abrimos con BOM UTF-8 y separador ';' (compatible con Excel en ES)
            self.csv_fp = open(csv_path, "w", newline="", encoding="utf-8-sig")
            self.csv_writer = csv.writer(self.csv_fp, delimiter=';')
            self.csv_writer.writerow([
                "timestamp", "evento", "clase",
                "car_in", "car_out", "moto_in", "moto_out",
                "car_inv", "moto_inv",
                "modelo", "conf", "orientacion", "pos_linea", "invertido", "fuente"
            ])
            self._csv_path_str = str(csv_path)
        except Exception as e:
            self._notify_error(f"No se pudo inicializar CSV: {e}")

    def _write_event_rows(
        self,
        car_in: int, car_out: int,
        moto_in: int, moto_out: int,
        car_inv: int, moto_inv: int,
    ) -> None:
        """Escribe filas por cada incremento observado de IN/OUT desde el último frame."""
        if not self.csv_writer:
            return
        now = datetime.now().isoformat(timespec="seconds")

        # Diferencias vs. último estado persistido
        di_car  = car_in  - self._prev_counts["car_in"]
        do_car  = car_out - self._prev_counts["car_out"]
        di_moto = moto_in - self._prev_counts["moto_in"]
        do_moto = moto_out- self._prev_counts["moto_out"]

        def _write_rows(n: int, evento: str, clase: str) -> None:
            for _ in range(max(0, n)):
                self.csv_writer.writerow([
                    now, evento, clase,
                    car_in, car_out, moto_in, moto_out,
                    car_inv, moto_inv,
                    self.config.model_name, f"{self.config.conf:.2f}",
                    self.config.line_orientation, f"{self.config.line_position:.2f}",
                    self.config.invert_direction,
                    "webcam" if isinstance(self.video_source, int) else str(self.video_source),
                ])

        _write_rows(di_car,  "IN",  "car")
        _write_rows(do_car,  "OUT", "car")
        _write_rows(di_moto, "IN",  "motorcycle")
        _write_rows(do_moto, "OUT", "motorcycle")

        # Actualizamos acumuladores
        self._prev_counts.update({
            "car_in": car_in, "car_out": car_out,
            "moto_in": moto_in, "moto_out": moto_out,
        })

    def _write_summary(self) -> None:
        """Escribe una fila final con totales e inventario al cerrar el stream."""
        if not self.csv_writer:
            return
        try:
            now = datetime.now().isoformat(timespec="seconds")
            self.csv_writer.writerow([
                now, "SUMMARY", "-",
                self._prev_counts["car_in"], self._prev_counts["car_out"],
                self._prev_counts["moto_in"], self._prev_counts["moto_out"],
                self._last_car_inv, self._last_moto_inv,
                self.config.model_name, f"{self.config.conf:.2f}",
                self.config.line_orientation, f"{self.config.line_position:.2f}",
                self.config.invert_direction,
                "webcam" if isinstance(self.video_source, int) else str(self.video_source),
            ])
        except Exception:
            pass

    # ---------- Loop principal ----------
    def run(self) -> None:
        """Bucle principal de procesamiento: captura → detecta → trackea → cuenta → (overlay/CSV)."""
        cap = None
        frame_start_time = None
        
        # Inicializar MLflow al inicio
        self._init_mlflow()
        
        try:
            # 1) Abrir fuente
            cap = cv2.VideoCapture(self.video_source)
            if not cap.isOpened():
                self._notify_error("No se pudo abrir el video/cámara.")
                return

            # Para progreso, intentamos obtener número de frames (si es archivo)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            sent_progress = -1

            # 2) Primer frame para validar y obtener dimensiones
            ok, frame = cap.read()
            if not ok or frame is None:
                self._notify_error("No se pudo leer el primer fotograma.")
                return

            h, w = frame.shape[:2]

            # 3) Definir línea de conteo según orientación/posición
            if self.config.line_orientation == "horizontal":
                y_pos = int(h * self.config.line_position)
                a = (0, y_pos)
                b = (w - 1, y_pos)
            else:
                x_pos = int(w * self.config.line_position)
                a = (x_pos, 0)
                b = (x_pos, h - 1)

            # 4) Inicializar CSV si está habilitado
            self._init_csv()

            # 5) Construir componentes: detector, tracker, contador y anotadores
            detector = VehicleDetector(
                model_name=self.config.model_name,
                conf=self.config.conf,
                iou=self.config.iou,
                device=self.config.device,
            )
            tracker = sv.ByteTrack()
            counter = LineCrossingCounterByClass(a=a, b=b, invert_direction=self.config.invert_direction)

            # Registrar información del modelo en MLflow
            self._log_model_info(detector)

            box_annotator = sv.BoxAnnotator()
            label_annotator = sv.LabelAnnotator()

            # 6) Ventana UI si display=True
            if self.display:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WINDOW_NAME, min(1280, w), min(720, h))

            pending_first = True  # ya tenemos el 1er frame leído
            processed = 0

            # 7) Bucle de frames
            while not self.stop_event.is_set():
                frame_start_time = time.time()
                
                if pending_first:
                    pending_first = False
                else:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break  # fin de video
                processed += 1

                # 7.1) Detección + tracking
                detections = detector.detect(frame)
                tracked = tracker.update_with_detections(detections)

                # 7.2) Etiquetas para overlay (clase, id y score)
                labels = []
                for i in range(len(tracked)):
                    cname = str(tracked.data.get("class_name", [""] * len(tracked))[i])
                    conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
                    tid = tracked.tracker_id[i] if tracked.tracker_id is not None else None
                    id_text = f"#{int(tid)}" if tid is not None else ""
                    labels.append(f"{cname} {id_text} {conf:.2f}")

                # 7.3) Overlay (opcional)
                draw_frame = frame.copy()
                draw_frame = box_annotator.annotate(scene=draw_frame, detections=tracked)
                draw_frame = label_annotator.annotate(scene=draw_frame, detections=tracked, labels=labels)

                # 7.4) Conteo por cruce
                counter.update(tracked)

                # 7.5) Dibujar línea y puntos extremos
                cv2.line(draw_frame, counter.a, counter.b, (0, 255, 255), 3)
                cv2.circle(draw_frame, counter.a, 5, (0, 255, 255), -1)
                cv2.circle(draw_frame, counter.b, 5, (0, 255, 255), -1)

                # 7.6) Panel de estado (conteos actuales e inventario)
                car_in = counter.in_counts.get("car", 0)
                car_out = counter.out_counts.get("car", 0)
                car_inv = counter.inventory.get("car", 0)
                moto_in = counter.in_counts.get("motorcycle", 0)
                moto_out = counter.out_counts.get("motorcycle", 0)
                moto_inv = counter.inventory.get("motorcycle", 0)

                # Dibujar panel de información en el video
                panel_w = 420
                panel_h = 140
                # Fondo negro semi-transparente para el panel
                overlay = draw_frame.copy()
                cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, draw_frame, 0.3, 0, draw_frame)
                
                # Título del panel
                cv2.putText(draw_frame, "Conteo IN/OUT e Inventario", (20, 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Información de carros
                cv2.putText(draw_frame, f"Carros -> IN: {car_in} | OUT: {car_out} | INV: {car_inv}/{self.config.capacity_car}",
                           (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 255, 80), 2)
                
                # Información de motos
                cv2.putText(draw_frame, f"Motos  -> IN: {moto_in} | OUT: {moto_out} | INV: {moto_inv}/{self.config.capacity_moto}",
                           (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 200, 255), 2)
                
                # Mensaje de salida en la esquina superior derecha
                exit_msg = "Presiona Q para salir"
                msg_size = cv2.getTextSize(exit_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                msg_x = draw_frame.shape[1] - msg_size[0] - 15
                msg_y = 30
                # Fondo para el mensaje
                cv2.rectangle(draw_frame, (msg_x - 5, msg_y - 20), (draw_frame.shape[1] - 5, msg_y + 5), (0, 0, 0), -1)
                cv2.putText(draw_frame, exit_msg, (msg_x, msg_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Guardar inventarios para SUMMARY final
                self._last_car_inv = car_inv
                self._last_moto_inv = moto_inv

                # 7.7) Alarma por capacidad
                over_car = car_inv > self.config.capacity_car
                over_moto = moto_inv > self.config.capacity_moto
                
                # Mostrar alerta visual si se excede la capacidad
                if over_car or over_moto:
                    alert_text = "ALERTA: CAPACIDAD EXCEDIDA"
                    # Texto parpadeante (usando el frame count para alternar)
                    if (processed // 15) % 2 == 0:  # Parpadea cada 15 frames
                        cv2.putText(draw_frame, alert_text, (20, 130), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        # Indicar qué tipo de vehículo excedió la capacidad
                        if over_car:
                            cv2.putText(draw_frame, "- CARROS EXCEDIDOS", (30, 155), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        if over_moto:
                            cv2.putText(draw_frame, "- MOTOS EXCEDIDAS", (30, 175), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Beep solo al momento de exceder, no constantemente
                if (over_car and not self._prev_over_car) or (over_moto and not self._prev_over_moto):
                    winsound_beep()
                    
                self._prev_over_car = over_car
                self._prev_over_moto = over_moto

                # 7.8) CSV por eventos (si está habilitado)
                self._write_event_rows(
                    car_in=car_in, car_out=car_out,
                    moto_in=moto_in, moto_out=moto_out,
                    car_inv=car_inv, moto_inv=moto_inv,
                )
                
                # 7.8b) MLflow: Registrar métricas de detección y conteo
                if frame_start_time:
                    frame_time = time.time() - frame_start_time
                    current_fps = 1.0 / frame_time if frame_time > 0 else 0
                else:
                    current_fps = None
                
                # Contar detecciones por clase
                detections_count = len(tracked)
                car_detections = sum(1 for i in range(len(tracked)) 
                                   if tracked.data.get("class_name", [""] * len(tracked))[i] == "car")
                moto_detections = sum(1 for i in range(len(tracked)) 
                                    if tracked.data.get("class_name", [""] * len(tracked))[i] == "motorcycle")
                
                self._log_detection_metrics(detections_count, car_detections, moto_detections, current_fps)
                self._log_counting_metrics(car_in, car_out, car_inv, moto_in, moto_out, moto_inv)
                    
                # Contar detecciones por clase
                car_detections = sum(1 for i in range(len(tracked)) 
                                   if tracked.data.get("class_name", [""])[i] == "car")
                moto_detections = sum(1 for i in range(len(tracked)) 
                                    if tracked.data.get("class_name", [""])[i] == "motorcycle")
                
                self._log_detection_metrics(
                    detections_count=len(tracked),
                    car_count=car_detections,
                    moto_count=moto_detections,
                    fps=current_fps
                )
                
                self._log_counting_metrics(
                    car_in=car_in, car_out=car_out, car_inv=car_inv,
                    moto_in=moto_in, moto_out=moto_out, moto_inv=moto_inv
                )

                # 7.9) Mostrar / callbacks
                if self.display:
                    cv2.imshow(WINDOW_NAME, draw_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.stop_event.set()
                        break
                else:
                    # Emitimos frame a la UI web
                    if self.on_frame is not None:
                        try:
                            rgb = cv2.cvtColor(draw_frame, cv2.COLOR_BGR2RGB)
                            self.on_frame(rgb)
                        except Exception:
                            pass

                # 7.10) Progreso
                if self.on_progress and total_frames > 0:
                    p = min(1.0, max(0.0, processed / float(total_frames)))
                    # Evitamos spamear demasiadas actualizaciones
                    step = int(p * 100)
                    if step != sent_progress:
                        sent_progress = step
                        try:
                            self.on_progress(p)
                        except Exception:
                            pass

        except Exception as e:
            self._notify_error(f"Fallo inesperado en el procesamiento: {e}")
        finally:
            # Liberación de recursos (siempre)
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass
            if self.display:
                try:
                    cv2.destroyWindow(WINDOW_NAME)
                    cv2.waitKey(1)
                    cv2.destroyAllWindows()
                except Exception:
                    pass

            # SUMMARY y cierre CSV
            try:
                self._write_summary()
            except Exception:
                pass
            finally:
                if self.csv_fp:
                    try:
                        self.csv_fp.close()
                    except Exception:
                        pass

            # Nuevas funcionalidades MLflow antes de finalizar
            if self.enable_mlflow and hasattr(self, '_log_system_info_called') == False:
                try:
                    # Usar la nueva integración MLflow si está disponible
                    from mlflow_integration import get_mlflow_tracker
                    tracker = get_mlflow_tracker()
                    
                    # Solo llamar una vez por run
                    if hasattr(tracker, 'log_system_information'):
                        tracker.log_system_information()
                    
                    if hasattr(tracker, 'create_and_log_visualizations'):
                        # Transferir datos del procesador al tracker
                        tracker.fps_samples = self.fps_samples
                        tracker.total_detections = self.detection_count
                        tracker.total_frames_processed = self.frame_count
                        tracker.start_time = self.mlflow_start_time
                        tracker.run_id = self.mlflow_run_id
                        tracker.create_and_log_visualizations()
                    
                    if hasattr(tracker, 'register_model_to_registry') and hasattr(detector, 'model'):
                        try:
                            tracker.register_model_to_registry(detector.model)
                        except Exception as e:
                            print(f"⚠️  Error registrando modelo en registry: {e}")
                    
                    self._log_system_info_called = True
                except Exception as e:
                    print(f"⚠️  Error en funcionalidades MLflow avanzadas: {e}")
            
            # Finalizar MLflow
            self._finish_mlflow()
            
            # Notificar fin al frontend
            if self.on_finish:
                try:
                    self.on_finish()
                except Exception:
                    pass
