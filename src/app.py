from __future__ import annotations
import os
import sys
import threading
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import cv2
import numpy as np
import supervision as sv
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from detector import VehicleDetector
from counter import LineCrossingCounterByClass


WINDOW_NAME = "Conteo de Vehículos - YOLOv11 + Supervision"


def _winsound_beep(freq: int = 1000, dur_ms: int = 250) -> None:
    if sys.platform.startswith("win"):
        try:
            import winsound
            winsound.Beep(freq, dur_ms)
        except Exception:
            pass


@dataclass
class AppConfig:
    model_name: str = "yolo11n.pt"  # Modelo ligero
    conf: float = 0.3
    iou: float = 0.5
    device: Optional[str] = None  # None -> auto (CPU/GPU)
    line_orientation: str = "vertical"    # "horizontal" | "vertical" - vertical por defecto
    line_position: float = 0.5            # Posición de la línea (0.0 a 1.0) - 0.5 = centro
    invert_direction: bool = False         # Invierte el sentido de IN/OUT
    capacity_car: int = 50
    capacity_moto: int = 50


class VideoProcessor(threading.Thread):
    def __init__(
        self,
        video_source: int | str,
        config: AppConfig,
        stop_event: threading.Event,
        on_error: callable | None = None,
        on_finish: callable | None = None,
    ) -> None:
        super().__init__(daemon=True)
        self.video_source = video_source
        self.config = config
        self.stop_event = stop_event
        self.on_error = on_error
        self.on_finish = on_finish
        self._prev_over_car = False
        self._prev_over_moto = False

    def _notify_error(self, msg: str) -> None:
        try:
            if self.on_error:
                self.on_error(msg)
            else:
                print(f"[ERROR] {msg}")
        except Exception:
            print(f"[ERROR] {msg}")

    def run(self) -> None:
        try:
            cap = cv2.VideoCapture(self.video_source)
            if not cap.isOpened():
                self._notify_error("No se pudo abrir el video/cámara.")
                return

            # Leer primer frame para conocer dimensiones
            ok, frame = cap.read()
            if not ok or frame is None:
                self._notify_error("No se pudo leer el primer fotograma.")
                cap.release()
                return

            h, w = frame.shape[:2]
            # Posicionar línea según configuración
            if self.config.line_orientation == "horizontal":
                # Línea horizontal: la posición afecta la altura (y)
                y_pos = int(h * self.config.line_position)
                a = (0, y_pos)
                b = (w - 1, y_pos)
            else:
                # Línea vertical: la posición afecta el ancho (x)
                x_pos = int(w * self.config.line_position)
                a = (x_pos, 0)
                b = (x_pos, h - 1)

            # Inicializar detector, tracker y contador
            detector = VehicleDetector(
                model_name=self.config.model_name,
                conf=self.config.conf,
                iou=self.config.iou,
                device=self.config.device,
            )
            tracker = sv.ByteTrack()
            counter = LineCrossingCounterByClass(a=a, b=b, invert_direction=self.config.invert_direction)

            # Anotadores - sin colores personalizados por ahora
            box_annotator = sv.BoxAnnotator()
            label_annotator = sv.LabelAnnotator()

            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, min(1280, w), min(720, h))

            # Como ya leímos el primer frame, procesémoslo primero
            pending_first = True

            while not self.stop_event.is_set():
                if pending_first:
                    pending_first = False
                else:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break

                detections = detector.detect(frame)
                tracked = tracker.update_with_detections(detections)

                # Etiquetas por detección
                labels = []
                for i in range(len(tracked)):
                    cname = str(tracked.data.get("class_name", [""] * len(tracked))[i])
                    conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
                    tid = tracked.tracker_id[i] if tracked.tracker_id is not None else None
                    id_text = f"#{int(tid)}" if tid is not None else ""
                    labels.append(f"{cname} {id_text} {conf:.2f}")

                # Dibujar cajas y etiquetas (sin colores personalizados por ahora)
                frame = box_annotator.annotate(scene=frame, detections=tracked)
                frame = label_annotator.annotate(scene=frame, detections=tracked, labels=labels)

                # Actualizar conteo por cruce de línea
                counter.update(tracked)

                # Dibujar la línea
                cv2.line(frame, counter.a, counter.b, (0, 255, 255), 3)
                cv2.circle(frame, counter.a, 5, (0, 255, 255), -1)
                cv2.circle(frame, counter.b, 5, (0, 255, 255), -1)

                # Panel de estado
                panel_w = 420
                cv2.rectangle(frame, (10, 10), (10 + panel_w, 140), (0, 0, 0), -1)
                cv2.putText(frame, "Conteo IN/OUT e Inventario", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Mensaje de salida en la esquina superior derecha
                exit_msg = "Presiona Q para salir"
                msg_size = cv2.getTextSize(exit_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                msg_x = frame.shape[1] - msg_size[0] - 15
                msg_y = 30
                # Fondo semi-transparente para el mensaje
                cv2.rectangle(frame, (msg_x - 5, msg_y - 20), (frame.shape[1] - 5, msg_y + 5), (0, 0, 0), -1)
                cv2.putText(frame, exit_msg, (msg_x, msg_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Carros
                car_in = counter.in_counts.get("car", 0)
                car_out = counter.out_counts.get("car", 0)
                car_inv = counter.inventory.get("car", 0)
                cv2.putText(frame, f"Carros -> IN: {car_in} | OUT: {car_out} | INV: {car_inv}/{self.config.capacity_car}",
                            (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 255, 80), 2)
                # Motos
                moto_in = counter.in_counts.get("motorcycle", 0)
                moto_out = counter.out_counts.get("motorcycle", 0)
                moto_inv = counter.inventory.get("motorcycle", 0)
                cv2.putText(frame, f"Motos  -> IN: {moto_in} | OUT: {moto_out} | INV: {moto_inv}/{self.config.capacity_moto}",
                            (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 200, 255), 2)

                # Alarma por capacidad
                over_car = car_inv > self.config.capacity_car
                over_moto = moto_inv > self.config.capacity_moto
                if over_car or over_moto:
                    cv2.putText(frame, "ALERTA: CAPACIDAD EXCEDIDA", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Beep solo al momento de exceder, no constantemente
                if (over_car and not self._prev_over_car) or (over_moto and not self._prev_over_moto):
                    _winsound_beep()
                self._prev_over_car = over_car
                self._prev_over_moto = over_moto

                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.stop_event.set()  # Señalar que debe detenerse
                    break

        except Exception as e:
            self._notify_error(f"Fallo inesperado en el procesamiento: {e}")
        finally:
            try:
                cap.release()
            except Exception:
                pass
            try:
                cv2.destroyWindow(WINDOW_NAME)
                cv2.waitKey(1)  # Procesar eventos pendientes de OpenCV
                cv2.destroyAllWindows()  # Asegurar que todas las ventanas se cierren
            except Exception:
                pass
            # Notificar al thread principal que hemos terminado
            if self.on_finish:
                try:
                    self.on_finish()
                except Exception:
                    pass


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Detección y Conteo de Vehículos")
        self.geometry("640x400")
        self.resizable(False, False)

        # Estado
        self.video_path: Optional[str] = None
        self.use_webcam = tk.BooleanVar(value=False)
        self.config = AppConfig()
        self._thread: Optional[VideoProcessor] = None
        self._stop_event = threading.Event()

        self._build_ui()

    def _build_ui(self) -> None:
        pad = {"padx": 10, "pady": 6}

        frm_src = ttk.LabelFrame(self, text="Fuente de video")
        frm_src.pack(fill=tk.X, **pad)

        chk_cam = ttk.Checkbutton(frm_src, text="Usar webcam (ID 0)", variable=self.use_webcam, command=self._on_toggle_source)
        chk_cam.pack(anchor=tk.W, **pad)

        src_row = ttk.Frame(frm_src)
        src_row.pack(fill=tk.X, **pad)
        self.lbl_video = ttk.Entry(src_row)
        self.lbl_video.configure(state="readonly")
        self.lbl_video.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        btn_browse = ttk.Button(src_row, text="Seleccionar video...", command=self._on_browse)
        btn_browse.pack(side=tk.LEFT)

        frm_cfg = ttk.LabelFrame(self, text="Configuración")
        frm_cfg.pack(fill=tk.X, **pad)

        # Modelo y umbrales
        row1 = ttk.Frame(frm_cfg)
        row1.pack(fill=tk.X, **pad)
        ttk.Label(row1, text="Modelo:").pack(side=tk.LEFT)
        self.cmb_model = ttk.Combobox(row1, values=["yolo11n.pt", "yolov8n.pt"], state="readonly")
        self.cmb_model.set(self.config.model_name)
        self.cmb_model.pack(side=tk.LEFT, padx=(6, 18))

        ttk.Label(row1, text="Conf:").pack(side=tk.LEFT)
        self.scale_conf = ttk.Scale(row1, from_=0.1, to=0.8, value=self.config.conf, command=lambda v: None)
        self.scale_conf.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 6))
        self.lbl_conf_val = ttk.Label(row1, text=f"{self.config.conf:.2f}")
        self.lbl_conf_val.pack(side=tk.LEFT)
        self.scale_conf.bind("<ButtonRelease-1>", lambda e: self._update_conf_label())

        row2 = ttk.Frame(frm_cfg)
        row2.pack(fill=tk.X, **pad)
        ttk.Label(row2, text="Orientación línea:").pack(side=tk.LEFT)
        self.cmb_orient = ttk.Combobox(row2, values=["horizontal", "vertical"], state="readonly")
        self.cmb_orient.set(self.config.line_orientation)
        self.cmb_orient.pack(side=tk.LEFT, padx=(6, 18))

        self.var_invert = tk.BooleanVar(value=self.config.invert_direction)
        chk_inv = ttk.Checkbutton(row2, text="Invertir dirección (IN<->OUT)", variable=self.var_invert)
        chk_inv.pack(side=tk.LEFT)
        
        # Slider para posición de línea
        row2b = ttk.Frame(frm_cfg)
        row2b.pack(fill=tk.X, **pad)
        ttk.Label(row2b, text="Posición línea:").pack(side=tk.LEFT)
        self.scale_line_pos = ttk.Scale(row2b, from_=0.1, to=0.9, value=self.config.line_position, command=lambda v: None)
        self.scale_line_pos.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 6))
        self.lbl_line_pos_val = ttk.Label(row2b, text=f"{int(self.config.line_position * 100)}%")
        self.lbl_line_pos_val.pack(side=tk.LEFT)
        self.scale_line_pos.bind("<ButtonRelease-1>", lambda e: self._update_line_pos_label())
        self.scale_line_pos.bind("<B1-Motion>", lambda e: self._update_line_pos_label())

        row3 = ttk.Frame(frm_cfg)
        row3.pack(fill=tk.X, **pad)
        ttk.Label(row3, text="Capacidad carros:").pack(side=tk.LEFT)
        self.sp_car = ttk.Spinbox(row3, from_=0, to=100000, width=8)
        self.sp_car.set(self.config.capacity_car)
        self.sp_car.pack(side=tk.LEFT, padx=(6, 18))

        ttk.Label(row3, text="Capacidad motos:").pack(side=tk.LEFT)
        self.sp_moto = ttk.Spinbox(row3, from_=0, to=100000, width=8)
        self.sp_moto.set(self.config.capacity_moto)
        self.sp_moto.pack(side=tk.LEFT)

        frm_run = ttk.Frame(self)
        frm_run.pack(fill=tk.X, **pad)

        self.btn_start = ttk.Button(frm_run, text="Iniciar", command=self._on_start)
        self.btn_start.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_stop = ttk.Button(frm_run, text="Detener", command=self._on_stop, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT)

        self.status = ttk.Label(self, text="Listo.")
        self.status.pack(fill=tk.X, **pad)

    def _update_conf_label(self) -> None:
        try:
            val = float(self.scale_conf.get())
        except Exception:
            val = 0.0
        self.lbl_conf_val.configure(text=f"{val:.2f}")
    
    def _update_line_pos_label(self) -> None:
        try:
            val = float(self.scale_line_pos.get())
        except Exception:
            val = 0.5
        self.lbl_line_pos_val.configure(text=f"{int(val * 100)}%")

    def _on_toggle_source(self) -> None:
        use_cam = self.use_webcam.get()
        state = "readonly" if not use_cam else tk.DISABLED
        self.lbl_video.configure(state=state)

    def _on_browse(self) -> None:
        path = filedialog.askopenfilename(
            title="Seleccionar video",
            filetypes=[
                ("Videos", "*.mp4;*.avi;*.mov;*.mkv"),
                ("Todos los archivos", "*.*"),
            ],
        )
        if path:
            self.video_path = path
            self.lbl_video.configure(state="normal")
            self.lbl_video.delete(0, tk.END)
            self.lbl_video.insert(0, path)
            self.lbl_video.configure(state="readonly")

    def _collect_config(self) -> Optional[Tuple[int | str, AppConfig]]:
        # Validar fuente
        source: int | str
        if self.use_webcam.get():
            source = 0
        else:
            if not self.video_path or not os.path.exists(self.video_path):
                messagebox.showwarning("Fuente", "Selecciona un archivo de video válido o marca webcam.")
                return None
            source = self.video_path

        # Configuración
        cfg = AppConfig(
            model_name=self.cmb_model.get(),
            conf=float(self.scale_conf.get()),
            iou=0.5,
            device=None,  # auto
            line_orientation=self.cmb_orient.get(),
            line_position=float(self.scale_line_pos.get()),
            invert_direction=self.var_invert.get(),
            capacity_car=int(self.sp_car.get()),
            capacity_moto=int(self.sp_moto.get()),
        )
        return source, cfg

    def _on_start(self) -> None:
        collected = self._collect_config()
        if not collected:
            return
        source, cfg = collected

        # Verificar si hay un thread anterior y esperar a que termine
        if self._thread and self._thread.is_alive():
            messagebox.showinfo("Ejecución", "La ejecución ya está en curso.")
            return
        
        # Si hay un thread anterior muerto, esperar un poco para limpieza
        if self._thread:
            self._thread.join(timeout=0.5)

        self._stop_event.clear()
        
        # Preparar callback de error hacia el hilo principal
        def on_err(msg: str) -> None:
            self.after(0, lambda: messagebox.showerror("Error", msg))
            self.after(0, lambda: self.status.configure(text=f"Error: {msg}"))
            self.after(0, lambda: self.btn_start.configure(state=tk.NORMAL))
            self.after(0, lambda: self.btn_stop.configure(state=tk.DISABLED))
        
        # Preparar callback de finalización
        def on_finish() -> None:
            self.after(0, lambda: self._handle_thread_finish())

        self._thread = VideoProcessor(
            video_source=source, 
            config=cfg, 
            stop_event=self._stop_event, 
            on_error=on_err,
            on_finish=on_finish
        )
        self._thread.start()
        self.status.configure(text="Procesando... (presiona 'q' en la ventana de video para detener)")
        self.btn_start.configure(state=tk.DISABLED)
        self.btn_stop.configure(state=tk.NORMAL)

    def _on_stop(self) -> None:
        self._stop_event.set()
        self.btn_stop.configure(state=tk.DISABLED)
        self.status.configure(text="Deteniendo...")
        # Esperar a que el thread termine antes de habilitar el botón de inicio
        if self._thread and self._thread.is_alive():
            self.after(100, self._check_thread_stopped)
        else:
            self.btn_start.configure(state=tk.NORMAL)
            self.status.configure(text="Detenido.")
    
    def _check_thread_stopped(self) -> None:
        """Verifica si el thread se ha detenido."""
        if self._thread and self._thread.is_alive():
            # Seguir esperando
            self.after(100, self._check_thread_stopped)
        else:
            self.btn_start.configure(state=tk.NORMAL)
            self.status.configure(text="Detenido.")
    
    def _handle_thread_finish(self) -> None:
        """Maneja la finalización del thread de procesamiento."""
        # Siempre actualizar el estado de los botones cuando el thread termina
        self.btn_stop.configure(state=tk.DISABLED)
        self.btn_start.configure(state=tk.NORMAL)
        self.status.configure(text="Detenido.")
        # Asegurar que el evento de parada esté configurado
        self._stop_event.set()


if __name__ == "__main__":
    app = App()
    app.mainloop()
