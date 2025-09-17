from __future__ import annotations
import os
import sys
import threading
from pathlib import Path
from typing import Optional, Tuple

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from config import AppConfig
from processor import VideoProcessor


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Detección y Conteo de Vehículos")
        self.geometry("680x480")
        self.resizable(False, False)

        # ---------------- Estado de la UI ----------------
        self.video_path: Optional[str] = None
        self.use_webcam = tk.BooleanVar(value=False)
        self.config_data = AppConfig()
        self._thread: Optional[VideoProcessor] = None
        self._stop_event = threading.Event()

        # Estado para controles de CSV
        self.var_enable_csv = tk.BooleanVar(value=self.config_data.enable_csv)
        self.var_csv_dir = tk.StringVar(value=self.config_data.csv_dir)
        self.var_csv_name = tk.StringVar(value=self.config_data.csv_name or "")

        self._build_ui()

    def _build_ui(self) -> None:
        """Construye todos los controles de la interfaz."""
        pad = {"padx": 10, "pady": 6}

        # ----- Fuente de video -----
        frm_src = ttk.LabelFrame(self, text="Fuente de video")
        frm_src.pack(fill=tk.X, **pad)

        chk_cam = ttk.Checkbutton(
            frm_src, text="Usar webcam (ID 0)",
            variable=self.use_webcam, command=self._on_toggle_source
        )
        chk_cam.pack(anchor=tk.W, **pad)

        src_row = ttk.Frame(frm_src)
        src_row.pack(fill=tk.X, **pad)
        self.lbl_video = ttk.Entry(src_row)
        self.lbl_video.configure(state="readonly")
        self.lbl_video.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        btn_browse = ttk.Button(src_row, text="Seleccionar video...", command=self._on_browse)
        btn_browse.pack(side=tk.LEFT)

        # ----- Configuración general -----
        frm_cfg = ttk.LabelFrame(self, text="Configuración")
        frm_cfg.pack(fill=tk.X, **pad)

        # Modelo + confianza
        row1 = ttk.Frame(frm_cfg)
        row1.pack(fill=tk.X, **pad)
        ttk.Label(row1, text="Modelo:").pack(side=tk.LEFT)
        self.cmb_model = ttk.Combobox(
            row1, values=["yolo11n.pt", "yolov8n.pt", "yolo12n.pt"], state="readonly"
        )
        self.cmb_model.set(self.config_data.model_name)
        self.cmb_model.pack(side=tk.LEFT, padx=(6, 18))

        ttk.Label(row1, text="Conf:").pack(side=tk.LEFT)
        self.scale_conf = ttk.Scale(
            row1, from_=0.1, to=0.8, value=self.config_data.conf, command=lambda v: None
        )
        self.scale_conf.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 6))
        self.lbl_conf_val = ttk.Label(row1, text=f"{self.config_data.conf:.2f}")
        self.lbl_conf_val.pack(side=tk.LEFT)
        self.scale_conf.bind("<ButtonRelease-1>", lambda e: self._update_conf_label())

        # Orientación de línea + invertir dirección
        row2 = ttk.Frame(frm_cfg)
        row2.pack(fill=tk.X, **pad)
        ttk.Label(row2, text="Orientación línea:").pack(side=tk.LEFT)
        self.cmb_orient = ttk.Combobox(
            row2, values=["horizontal", "vertical"], state="readonly"
        )
        self.cmb_orient.set(self.config_data.line_orientation)
        self.cmb_orient.pack(side=tk.LEFT, padx=(6, 18))

        self.var_invert = tk.BooleanVar(value=self.config_data.invert_direction)
        chk_inv = ttk.Checkbutton(
            row2, text="Invertir dirección (IN<->OUT)", variable=self.var_invert
        )
        chk_inv.pack(side=tk.LEFT)

        # Posición de línea (slider)
        row2b = ttk.Frame(frm_cfg)
        row2b.pack(fill=tk.X, **pad)
        ttk.Label(row2b, text="Posición línea:").pack(side=tk.LEFT)
        self.scale_line_pos = ttk.Scale(
            row2b, from_=0.1, to=0.9, value=self.config_data.line_position, command=lambda v: None
        )
        self.scale_line_pos.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 6))
        self.lbl_line_pos_val = ttk.Label(
            row2b, text=f"{int(self.config_data.line_position * 100)}%"
        )
        self.lbl_line_pos_val.pack(side=tk.LEFT)
        self.scale_line_pos.bind("<ButtonRelease-1>", lambda e: self._update_line_pos_label())
        self.scale_line_pos.bind("<B1-Motion>", lambda e: self._update_line_pos_label())

        # Capacidades
        row3 = ttk.Frame(frm_cfg)
        row3.pack(fill=tk.X, **pad)
        ttk.Label(row3, text="Capacidad carros:").pack(side=tk.LEFT)
        self.sp_car = ttk.Spinbox(row3, from_=0, to=100000, width=8)
        self.sp_car.set(self.config_data.capacity_car)
        self.sp_car.pack(side=tk.LEFT, padx=(6, 18))

        ttk.Label(row3, text="Capacidad motos:").pack(side=tk.LEFT)
        self.sp_moto = ttk.Spinbox(row3, from_=0, to=100000, width=8)
        self.sp_moto.set(self.config_data.capacity_moto)
        self.sp_moto.pack(side=tk.LEFT)

        # ----- Controles de CSV -----
        frm_csv = ttk.LabelFrame(self, text="Reporte CSV")
        frm_csv.pack(fill=tk.X, **pad)

        chk_csv = ttk.Checkbutton(
            frm_csv, text="Guardar CSV de eventos", variable=self.var_enable_csv
        )
        chk_csv.pack(anchor=tk.W, **pad)

        row_csv = ttk.Frame(frm_csv)
        row_csv.pack(fill=tk.X, **pad)
        ttk.Label(row_csv, text="Carpeta:").pack(side=tk.LEFT)
        self.ent_csv_dir = ttk.Entry(row_csv, textvariable=self.var_csv_dir)
        self.ent_csv_dir.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 6))
        btn_csv_dir = ttk.Button(row_csv, text="Seleccionar...", command=self._on_pick_csv_dir)
        btn_csv_dir.pack(side=tk.LEFT)

        row_csv_name = ttk.Frame(frm_csv)
        row_csv_name.pack(fill=tk.X, **pad)
        ttk.Label(row_csv_name, text="Nombre archivo:").pack(side=tk.LEFT)
        hint = ttk.Label(row_csv_name, text="(opcional, sin .csv)", foreground="#888")
        hint.pack(side=tk.RIGHT, padx=(6, 0))
        self.ent_csv_name = ttk.Entry(row_csv_name, textvariable=self.var_csv_name)
        self.ent_csv_name.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 6))

        # ----- Acciones -----
        frm_run = ttk.Frame(self)
        frm_run.pack(fill=tk.X, **pad)

        self.btn_start = ttk.Button(frm_run, text="Iniciar (UI)", command=self._on_start)
        self.btn_start.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_stop = ttk.Button(frm_run, text="Detener", command=self._on_stop, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_copy_cli = ttk.Button(
            frm_run, text="Copiar comando CLI (headless)", command=self._on_copy_cli
        )
        self.btn_copy_cli.pack(side=tk.LEFT)

        self.status = ttk.Label(self, text="Listo.")
        self.status.pack(fill=tk.X, **pad)

    # ----- Handlers UI -----
    def _update_conf_label(self) -> None:
        """Actualiza el texto del label de confianza con el valor del slider."""
        try:
            val = float(self.scale_conf.get())
        except Exception:
            val = 0.0
        self.lbl_conf_val.configure(text=f"{val:.2f}")

    def _update_line_pos_label(self) -> None:
        """Muestra la posición de línea en porcentaje mientras mueves el slider."""
        try:
            val = float(self.scale_line_pos.get())
        except Exception:
            val = 0.5
        self.lbl_line_pos_val.configure(text=f"{int(val * 100)}%")

    def _on_toggle_source(self) -> None:
        """Habilita/deshabilita el campo de ruta según se use webcam o archivo."""
        use_cam = self.use_webcam.get()
        state = "readonly" if not use_cam else tk.DISABLED
        self.lbl_video.configure(state=state)

    def _on_browse(self) -> None:
        """Abre el diálogo para seleccionar archivo de video."""
        path = filedialog.askopenfilename(
            title="Seleccionar video",
            filetypes=[("Videos", "*.mp4;*.avi;*.mov;*.mkv"), ("Todos los archivos", "*.*")],
        )
        if path:
            self.video_path = path
            self.lbl_video.configure(state="normal")
            self.lbl_video.delete(0, tk.END)
            self.lbl_video.insert(0, path)
            self.lbl_video.configure(state="readonly")

    def _on_pick_csv_dir(self) -> None:
        """Selecciona carpeta donde se guardarán los reportes CSV."""
        d = filedialog.askdirectory(title="Seleccionar carpeta para CSV")
        if d:
            self.var_csv_dir.set(d)

    def _collect_config(self) -> Optional[Tuple[int | str, AppConfig]]:
        """
        Construye un AppConfig con los valores actuales de la UI y devuelve (source, cfg).
        Valida que exista la ruta del video si no se usa webcam.
        """
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
        """Crea el VideoProcessor en un hilo y arranca el pipeline."""
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
            # Callbacks de UI para errores y finalización
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
        """Solicita detener el procesamiento y restaura los botones cuando termina."""
        self._stop_event.set()
        self.btn_stop.configure(state=tk.DISABLED)
        self.status.configure(text="Deteniendo...")
        if self._thread and self._thread.is_alive():
            self.after(100, self._check_thread_stopped)
        else:
            self.btn_start.configure(state=tk.NORMAL)
            self.status.configure(text="Detenido.")

    def _check_thread_stopped(self) -> None:
        """Sondea el hilo hasta que termine y actualiza el estado de la UI."""
        if self._thread and self._thread.is_alive():
            self.after(100, self._check_thread_stopped)
        else:
            self.btn_start.configure(state=tk.NORMAL)
            self.status.configure(text="Detenido.")

    def _handle_thread_finish(self) -> None:
        """Callback al finalizar procesamiento: limpia estados de UI."""
        self.btn_stop.configure(state=tk.DISABLED)
        self.btn_start.configure(state=tk.NORMAL)
        self.status.configure(text="Detenido.")
        self._stop_event.set()

    def _on_copy_cli(self) -> None:
        """Genera y copia al portapapeles el comando CLI equivalente (modo headless)."""
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
            f'"{py}" "{app_path.parent / "app.py"}" --cli {src_part} '
            f'--model "{cfg.model_name}" --conf {cfg.conf:.2f} '
            f'--orientation {cfg.line_orientation} --line-pos {cfg.line_position:.2f} '
            f'{"--invert" if cfg.invert_direction else ""} '
            f'--cap-car {cfg.capacity_car} --cap-moto {cfg.capacity_moto} '
            f'{"--csv" if cfg.enable_csv else "--no-csv"} '
            f'--csv-dir "{cfg.csv_dir}" '
            f'{name_flag}'
            f'--no-display'
        )

        try:
            self.clipboard_clear()
            self.clipboard_append(cmd)
            messagebox.showinfo("CLI copiado", "Se copió al portapapeles el comando para correr headless.")
        except Exception:
            # Si el clipboard no está disponible, mostramos el comando en la barra de estado
            self.status.configure(text="Comando CLI: " + cmd)
