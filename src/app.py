from __future__ import annotations
import os
import sys
import csv
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

from datetime import datetime
from pathlib import Path
import argparse

import cv2
import numpy as np
import supervision as sv
TK_AVAILABLE = False
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    TK_AVAILABLE = True
except Exception:
    tk = ttk = filedialog = messagebox = None

from detector import VehicleDetector
from counter import LineCrossingCounterByClass

from dataclasses import dataclass
from typing import Optional


WINDOW_NAME = "Conteo de Vehículos - YOLOv11/12 + Supervision"


def _winsound_beep(freq: int = 1000, dur_ms: int = 250) -> None:
    if sys.platform.startswith("win"):
        try:
            import winsound
            winsound.Beep(freq, dur_ms)
        except Exception:
            pass


@dataclass
class AppConfig:
    model_name: str = "yolo11n.pt"  # "yolo12n.pt", "yolo11n.pt", "yolov8n.pt"
    conf: float = 0.3
    iou: float = 0.5
    device: Optional[str] = None  # None -> auto (CPU/GPU)
    line_orientation: str = "vertical"    # "horizontal" | "vertical"
    line_position: float = 0.5            # 0.1 a 0.9 (UI); 0.5 = centro
    invert_direction: bool = False        # Invierte IN/OUT
    capacity_car: int = 50
    capacity_moto: int = 50

    # Reporte CSV
    enable_csv: bool = True
    csv_dir: str = "reports"
    csv_name: Optional[str] = ""   # << NUEVO: nombre opcional del archivo CSV (sin ruta)

def _sanitize_filename(name: str) -> str:
    """
    Reemplaza caracteres no válidos en Windows/macOS/Linux.
    Si queda vacío, retorna 'reporte'.
    """
    invalid = '<>:"/\\|?*'
    cleaned = "".join(('_' if ch in invalid else ch) for ch in name).strip()
    return cleaned or "reporte"

class VideoProcessor(threading.Thread):
    """
    Procesa un stream de video aplicando detección, tracking y conteo.
    Si display=False, corre en modo headless (sin ventanas), ideal para CLI.
    """
    def __init__(
        self,
        video_source: int | str,
        config: AppConfig,
        stop_event: threading.Event,
        on_error: callable | None = None,
        on_finish: callable | None = None,
        display: bool = True,
    ) -> None:
        super().__init__(daemon=True)
        self.video_source = video_source
        self.config = config
        self.stop_event = stop_event
        self.on_error = on_error
        self.on_finish = on_finish
        self.display = display

        self._prev_over_car = False
        self._prev_over_moto = False

        # CSV y acumuladores
        self.csv_fp = None
        self.csv_writer = None
        self._prev_counts = {"car_in": 0, "car_out": 0, "moto_in": 0, "moto_out": 0}
        self._last_car_inv = 0
        self._last_moto_inv = 0
        self._csv_path_str = None  # por si quieres mostrarlo

    # ---------- Utilidades internas ----------
    def _notify_error(self, msg: str) -> None:
        try:
            if self.on_error:
                self.on_error(msg)
            else:
                print(f"[ERROR] {msg}")
        except Exception:
            print(f"[ERROR] {msg}")

    def _init_csv(self) -> None:
        if not getattr(self.config, "enable_csv", False):
            return
        try:
            Path(self.config.csv_dir).mkdir(parents=True, exist_ok=True)
            ts_name = datetime.now().strftime("%Y%m%d_%H%M%S")
            src_name = "webcam" if isinstance(self.video_source, int) else Path(str(self.video_source)).name

            # << NUEVO: respeta nombre si el usuario lo dio; si no, usa el automático
            if getattr(self.config, "csv_name", None):
                base = _sanitize_filename(str(self.config.csv_name))
                filename = base if base.lower().endswith(".csv") else f"{base}.csv"
            else:
                filename = f"reporte_{ts_name}_{src_name}.csv"

            csv_path = Path(self.config.csv_dir) / filename

            # Evitar sobrescritura accidental: si existe, agrega sufijo _1, _2, ...
            if csv_path.exists():
                stem = csv_path.stem
                suffix = 1
                while True:
                    candidate = csv_path.with_name(f"{stem}_{suffix}.csv")
                    if not candidate.exists():
                        csv_path = candidate
                        break
                    suffix += 1

            # BOM + delimitador ; (Excel ES)
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
        """Escribe filas por cada incremento de IN/OUT detectado."""
        if not self.csv_writer:
            return
        now = datetime.now().isoformat(timespec="seconds")

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

        self._prev_counts.update({
            "car_in": car_in, "car_out": car_out,
            "moto_in": moto_in, "moto_out": moto_out,
        })

    def _write_summary(self) -> None:
        """Escribe una fila final de resumen."""
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
        try:
            cap = cv2.VideoCapture(self.video_source)
            if not cap.isOpened():
                self._notify_error("No se pudo abrir el video/cámara.")
                return

            # Primer frame para dimensiones
            ok, frame = cap.read()
            if not ok or frame is None:
                self._notify_error("No se pudo leer el primer fotograma.")
                cap.release()
                return

            h, w = frame.shape[:2]
            # Línea según configuración
            if self.config.line_orientation == "horizontal":
                y_pos = int(h * self.config.line_position)
                a = (0, y_pos)
                b = (w - 1, y_pos)
            else:
                x_pos = int(w * self.config.line_position)
                a = (x_pos, 0)
                b = (x_pos, h - 1)

            # CSV
            self._init_csv()

            # Detector, tracker, contador
            detector = VehicleDetector(
                model_name=self.config.model_name,
                conf=self.config.conf,
                iou=self.config.iou,
                device=self.config.device,
            )
            tracker = sv.ByteTrack()
            counter = LineCrossingCounterByClass(a=a, b=b, invert_direction=self.config.invert_direction)

            box_annotator = sv.BoxAnnotator()
            label_annotator = sv.LabelAnnotator()

            if self.display:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WINDOW_NAME, min(1280, w), min(720, h))

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

                # Etiquetas
                labels = []
                for i in range(len(tracked)):
                    cname = str(tracked.data.get("class_name", [""] * len(tracked))[i])
                    conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
                    tid = tracked.tracker_id[i] if tracked.tracker_id is not None else None
                    id_text = f"#{int(tid)}" if tid is not None else ""
                    labels.append(f"{cname} {id_text} {conf:.2f}")

                # Dibujo
                if self.display:
                    frame = box_annotator.annotate(scene=frame, detections=tracked)
                    frame = label_annotator.annotate(scene=frame, detections=tracked, labels=labels)

                # Conteo
                counter.update(tracked)

                # Línea
                if self.display:
                    cv2.line(frame, counter.a, counter.b, (0, 255, 255), 3)
                    cv2.circle(frame, counter.a, 5, (0, 255, 255), -1)
                    cv2.circle(frame, counter.b, 5, (0, 255, 255), -1)

                # Panel de estado
                car_in = counter.in_counts.get("car", 0)
                car_out = counter.out_counts.get("car", 0)
                car_inv = counter.inventory.get("car", 0)
                moto_in = counter.in_counts.get("motorcycle", 0)
                moto_out = counter.out_counts.get("motorcycle", 0)
                moto_inv = counter.inventory.get("motorcycle", 0)

                # Guardar inventarios para SUMMARY
                self._last_car_inv = car_inv
                self._last_moto_inv = moto_inv

                if self.display:
                    panel_w = 420
                    cv2.rectangle(frame, (10, 10), (10 + panel_w, 140), (0, 0, 0), -1)
                    cv2.putText(frame, "Conteo IN/OUT e Inventario", (20, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    exit_msg = "Presiona Q para salir"
                    msg_size = cv2.getTextSize(exit_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    msg_x = frame.shape[1] - msg_size[0] - 15
                    msg_y = 30
                    cv2.rectangle(frame, (msg_x - 5, msg_y - 20), (frame.shape[1] - 5, msg_y + 5), (0, 0, 0), -1)
                    cv2.putText(frame, exit_msg, (msg_x, msg_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    cv2.putText(
                        frame,
                        f"Carros -> IN: {car_in} | OUT: {car_out} | INV: {car_inv}/{self.config.capacity_car}",
                        (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 255, 80), 2
                    )
                    cv2.putText(
                        frame,
                        f"Motos  -> IN: {moto_in} | OUT: {moto_out} | INV: {moto_inv}/{self.config.capacity_moto}",
                        (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 200, 255), 2
                    )

                # Alarma por capacidad
                over_car = car_inv > self.config.capacity_car
                over_moto = moto_inv > self.config.capacity_moto
                if self.display and (over_car or over_moto):
                    cv2.putText(frame, "ALERTA: CAPACIDAD EXCEDIDA", (20, 130),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if (over_car and not self._prev_over_car) or (over_moto and not self._prev_over_moto):
                    _winsound_beep()
                self._prev_over_car = over_car
                self._prev_over_moto = over_moto

                # CSV por eventos
                self._write_event_rows(
                    car_in=car_in, car_out=car_out,
                    moto_in=moto_in, moto_out=moto_out,
                    car_inv=car_inv, moto_inv=moto_inv,
                )

                # Mostrar / teclado
                if self.display:
                    cv2.imshow(WINDOW_NAME, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.stop_event.set()
                        break

        except Exception as e:
            self._notify_error(f"Fallo inesperado en el procesamiento: {e}")
        finally:
            try:
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

            # Notificar fin
            if self.on_finish:
                try:
                    self.on_finish()
                except Exception:
                    pass


# =========================
#      A P P  ( T K )
# =========================
if TK_AVAILABLE:
    class App(tk.Tk):
        def __init__(self) -> None:
            super().__init__()
            self.title("Detección y Conteo de Vehículos")
            self.geometry("680x480")
            self.resizable(False, False)

            # Estado
            self.video_path: Optional[str] = None
            self.use_webcam = tk.BooleanVar(value=False)
            self.config = AppConfig()
            self._thread: Optional[VideoProcessor] = None
            self._stop_event = threading.Event()

            # UI state (CSV)
            self.var_enable_csv = tk.BooleanVar(value=self.config.enable_csv)
            self.var_csv_dir = tk.StringVar(value=self.config.csv_dir)
            self.var_csv_name = tk.StringVar(value=self.config.csv_name or "")   # << NUEVO

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
            self.cmb_model = ttk.Combobox(row1, values=["yolo11n.pt", "yolov8n.pt", "yolo12n.pt"], state="readonly")
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

            # Slider posición de línea
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

            # ---- CSV controls ----
            frm_csv = ttk.LabelFrame(self, text="Reporte CSV")
            frm_csv.pack(fill=tk.X, **pad)

            chk_csv = ttk.Checkbutton(frm_csv, text="Guardar CSV de eventos", variable=self.var_enable_csv)
            chk_csv.pack(anchor=tk.W, **pad)

            row_csv = ttk.Frame(frm_csv)
            row_csv.pack(fill=tk.X, **pad)
            ttk.Label(row_csv, text="Carpeta:").pack(side=tk.LEFT)
            self.ent_csv_dir = ttk.Entry(row_csv, textvariable=self.var_csv_dir)
            self.ent_csv_dir.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 6))
            btn_csv_dir = ttk.Button(row_csv, text="Seleccionar...", command=self._on_pick_csv_dir)
            btn_csv_dir.pack(side=tk.LEFT)

            # << NUEVO: nombre de archivo (opcional)
            row_csv_name = ttk.Frame(frm_csv)
            row_csv_name.pack(fill=tk.X, **pad)
            ttk.Label(row_csv_name, text="Nombre archivo:").pack(side=tk.LEFT)
            hint = ttk.Label(row_csv_name, text="(opcional, sin .csv)", foreground="#888")
            hint.pack(side=tk.RIGHT, padx=(6, 0))
            self.ent_csv_name = ttk.Entry(row_csv_name, textvariable=self.var_csv_name)
            self.ent_csv_name.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 6))


            # ---- Acciones ----
            frm_run = ttk.Frame(self)
            frm_run.pack(fill=tk.X, **pad)

            self.btn_start = ttk.Button(frm_run, text="Iniciar (UI)", command=self._on_start)
            self.btn_start.pack(side=tk.LEFT, padx=(0, 6))

            self.btn_stop = ttk.Button(frm_run, text="Detener", command=self._on_stop, state=tk.DISABLED)
            self.btn_stop.pack(side=tk.LEFT, padx=(0, 6))

            self.btn_copy_cli = ttk.Button(frm_run, text="Copiar comando CLI (headless)", command=self._on_copy_cli)
            self.btn_copy_cli.pack(side=tk.LEFT)

            self.status = ttk.Label(self, text="Listo.")
            self.status.pack(fill=tk.X, **pad)

        # ----- Handlers UI -----
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

        def _on_pick_csv_dir(self) -> None:
            d = filedialog.askdirectory(title="Seleccionar carpeta para CSV")
            if d:
                self.var_csv_dir.set(d)

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
                enable_csv=bool(self.var_enable_csv.get()),
                csv_dir=self.var_csv_dir.get() or "reports",
                csv_name=(self.var_csv_name.get().strip() or ""), 
            )
            return source, cfg

        def _on_start(self) -> None:
            collected = self._collect_config()
            if not collected:
                return
            source, cfg = collected

            if self._thread and self._thread.is_alive():
                messagebox.showinfo("Ejecución", "La ejecución ya está en curso.")
                return

            if self._thread:
                self._thread.join(timeout=0.5)

            self._stop_event.clear()

            def on_err(msg: str) -> None:
                self.after(0, lambda: messagebox.showerror("Error", msg))
                self.after(0, lambda: self.status.configure(text=f"Error: {msg}"))
                self.after(0, lambda: self.btn_start.configure(state=tk.NORMAL))
                self.after(0, lambda: self.btn_stop.configure(state=tk.DISABLED))

            def on_finish() -> None:
                self.after(0, lambda: self._handle_thread_finish())

            self._thread = VideoProcessor(
                video_source=source,
                config=cfg,
                stop_event=self._stop_event,
                on_error=on_err,
                on_finish=on_finish,
                display=True,  # UI = con ventanas
            )
            self._thread.start()
            self.status.configure(text="Procesando... (presiona 'q' en la ventana de video para detener)")
            self.btn_start.configure(state=tk.DISABLED)
            self.btn_stop.configure(state=tk.NORMAL)

        def _on_stop(self) -> None:
            self._stop_event.set()
            self.btn_stop.configure(state=tk.DISABLED)
            self.status.configure(text="Deteniendo...")
            if self._thread and self._thread.is_alive():
                self.after(100, self._check_thread_stopped)
            else:
                self.btn_start.configure(state=tk.NORMAL)
                self.status.configure(text="Detenido.")

        def _check_thread_stopped(self) -> None:
            if self._thread and self._thread.is_alive():
                self.after(100, self._check_thread_stopped)
            else:
                self.btn_start.configure(state=tk.NORMAL)
                self.status.configure(text="Detenido.")

        def _handle_thread_finish(self) -> None:
            self.btn_stop.configure(state=tk.DISABLED)
            self.btn_start.configure(state=tk.NORMAL)
            self.status.configure(text="Detenido.")
            self._stop_event.set()

        def _on_copy_cli(self) -> None:
            collected = self._collect_config()
            if not collected:
                return
            source, cfg = collected

            # Construir comando CLI equivalente (headless, sin display)
            py = sys.executable or "python"
            app_path = Path(__file__).resolve()
            if isinstance(source, int):
                src_part = "--webcam"
            else:
                src_part = f'--source "{str(source)}"'

            name_flag = f'--csv-name "{cfg.csv_name}" ' if (cfg.csv_name or "").strip() else ""
            cmd = (
                f'"{py}" "{app_path}" --cli {src_part} '
                f'--model "{cfg.model_name}" --conf {cfg.conf:.2f} '
                f'--orientation {cfg.line_orientation} --line-pos {cfg.line_position:.2f} '
                f'{"--invert" if cfg.invert_direction else ""} '
                f'--cap-car {cfg.capacity_car} --cap-moto {cfg.capacity_moto} '
                f'{"--csv" if cfg.enable_csv else "--no-csv"} '
                f'--csv-dir "{cfg.csv_dir}" '
                f'{name_flag}'
                f'--no-display'
            )

            # Copiar al portapapeles
            try:
                self.clipboard_clear()
                self.clipboard_append(cmd)
                messagebox.showinfo("CLI copiado", "Se copió al portapapeles el comando para correr headless.")
            except Exception:
                self.status.configure(text="Comando CLI: " + cmd)

# =========================
#        C L I
# =========================
def parse_cli_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Conteo de vehículos (headless/CLI)")
    p.add_argument("--cli", action="store_true", help="Usar modo CLI (sin UI)")
    src_grp = p.add_mutually_exclusive_group()
    src_grp.add_argument("--source", type=str, help="Ruta a video (mp4/avi/...)", default=None)
    src_grp.add_argument("--webcam", action="store_true", help="Usar webcam (ID 0)")

    p.add_argument("--model", type=str, default="yolo11n.pt", help="Modelo YOLO (yolo12n.pt|yolo11n.pt|yolov8n.pt)")
    p.add_argument("--conf", type=float, default=0.3, help="Confianza (0.1-0.8)")
    p.add_argument("--orientation", choices=["horizontal", "vertical"], default="vertical", help="Orientación de línea")
    p.add_argument("--line-pos", type=float, default=0.5, help="Posición de línea [0.1-0.9]")
    p.add_argument("--invert", action="store_true", help="Invertir dirección IN/OUT")
    p.add_argument("--cap-car", type=int, default=50, help="Capacidad carros")
    p.add_argument("--cap-moto", type=int, default=50, help="Capacidad motos")
    p.add_argument("--csv", dest="csv", action="store_true", help="Guardar CSV de eventos")
    p.add_argument("--no-csv", dest="csv", action="store_false", help="No guardar CSV")
    p.set_defaults(csv=True)
    p.add_argument("--csv-dir", type=str, default="reports", help="Carpeta para CSV")
    p.add_argument("--csv-name", type=str, default="", help="Nombre del archivo CSV (opcional, sin ruta)")
    dis = p.add_mutually_exclusive_group()
    dis.add_argument("--display", action="store_true", help="Mostrar ventanas (no recomendado en CLI)")
    dis.add_argument("--no-display", action="store_true", help="Sin ventanas (headless)")
    return p.parse_args(argv)


def main_cli(ns: argparse.Namespace) -> int:
    # Fuente
    if ns.webcam:
        source = 0
    elif ns.source:
        source = ns.source
        if not os.path.exists(source):
            print(f"[ERROR] Archivo no encontrado: {source}")
            return 2
    else:
        print("[ERROR] Debes indicar --webcam o --source <video>")
        return 2

    cfg = AppConfig(
        model_name=ns.model,
        conf=float(ns.conf),
        iou=0.5,
        device=None,
        line_orientation=ns.orientation,
        line_position=float(ns.line_pos),
        invert_direction=bool(ns.invert),
        capacity_car=int(ns.cap_car),
        capacity_moto=int(ns.cap_moto),
        enable_csv=bool(ns.csv),
        csv_dir=str(ns.csv_dir),
        csv_name=str(ns.csv_name or ""),
    )

    stop_event = threading.Event()
    vp = VideoProcessor(
        video_source=source,
        config=cfg,
        stop_event=stop_event,
        on_error=lambda m: print("[ERROR]", m),
        on_finish=None,
        display=bool(ns.display and not ns.no_display) if (ns.display or ns.no_display) else False,
    )

    # En CLI lo ejecutamos sin hilos (sincronamente)
    vp.run()
    return 0


if __name__ == "__main__":
    if any(a in sys.argv for a in ["--cli", "--source", "--webcam"]):
        # Modo CLI: no requiere Tk
        exit(main_cli(parse_cli_args(sys.argv[1:])))
    else:
        if not TK_AVAILABLE:
            print("Tkinter no está disponible en este entorno. Ejecuta en modo CLI: python src/app.py --cli ...")
            sys.exit(2)
        app = App()
        app.mainloop()



