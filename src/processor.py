from __future__ import annotations
import os
import csv
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import supervision as sv

from detector import VehicleDetector
from counter import LineCrossingCounterByClass
from config import AppConfig, WINDOW_NAME, sanitize_filename
from utils import winsound_beep


class VideoProcessor(threading.Thread):
    """
    Procesa un stream de video aplicando detección, tracking y conteo.
    Si display=False, corre en modo headless (sin ventanas), ideal para CLI.
    """
    def __init__(
        self,
        video_source: int | str,              # 0 (webcam) o ruta a video
        config: AppConfig,                    # parámetros de detección/contador/CSV
        stop_event: threading.Event,          # bandera para interrupción segura
        on_error: callable | None = None,     # callback de error (UI/CLI)
        on_finish: callable | None = None,    # callback al terminar
        display: bool = True,                 # True=con ventanas; False=headless
    ) -> None:
        super().__init__(daemon=True)
        self.video_source = video_source
        self.config = config
        self.stop_event = stop_event
        self.on_error = on_error
        self.on_finish = on_finish
        self.display = display

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
        try:
            if self.on_error:
                self.on_error(msg)
            else:
                print(f"[ERROR] {msg}")
        except Exception:
            print(f"[ERROR] {msg}")

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
        """
        Escribe filas por cada incremento observado de IN/OUT desde el último frame.
        Esto hace que el CSV sea “por evento”, no por frame.
        """
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

        # Emitimos una fila por cada incremento detectado
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
            # El summary es informativo; no interrumpimos si falla la escritura
            pass

    # ---------- Loop principal ----------
    def run(self) -> None:
        """Bucle principal de procesamiento: captura → detecta → trackea → cuenta → (overlay/CSV)."""
        try:
            # 1) Abrir fuente
            cap = cv2.VideoCapture(self.video_source)
            if not cap.isOpened():
                self._notify_error("No se pudo abrir el video/cámara.")
                return

            # 2) Primer frame para validar y obtener dimensiones
            ok, frame = cap.read()
            if not ok or frame is None:
                self._notify_error("No se pudo leer el primer fotograma.")
                cap.release()
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

            box_annotator = sv.BoxAnnotator()
            label_annotator = sv.LabelAnnotator()

            # 6) Ventana UI si display=True
            if self.display:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WINDOW_NAME, min(1280, w), min(720, h))

            pending_first = True  # ya tenemos el 1er frame leído

            # 7) Bucle de frames
            while not self.stop_event.is_set():
                if pending_first:
                    pending_first = False
                else:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break  # fin de video

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
                if self.display:
                    frame = box_annotator.annotate(scene=frame, detections=tracked)
                    frame = label_annotator.annotate(scene=frame, detections=tracked, labels=labels)

                # 7.4) Conteo por cruce
                counter.update(tracked)

                # 7.5) Dibujar línea y puntos extremos
                if self.display:
                    cv2.line(frame, counter.a, counter.b, (0, 255, 255), 3)
                    cv2.circle(frame, counter.a, 5, (0, 255, 255), -1)
                    cv2.circle(frame, counter.b, 5, (0, 255, 255), -1)

                # 7.6) Panel de estado (conteos actuales e inventario)
                car_in = counter.in_counts.get("car", 0)
                car_out = counter.out_counts.get("car", 0)
                car_inv = counter.inventory.get("car", 0)
                moto_in = counter.in_counts.get("motorcycle", 0)
                moto_out = counter.out_counts.get("motorcycle", 0)
                moto_inv = counter.inventory.get("motorcycle", 0)

                # Guardar inventarios para SUMMARY final
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

                # 7.7) Alarma por capacidad (beep solo cuando cruza el umbral)
                over_car = car_inv > self.config.capacity_car
                over_moto = moto_inv > self.config.capacity_moto
                if self.display and (over_car or over_moto):
                    cv2.putText(frame, "ALERTA: CAPACIDAD EXCEDIDA", (20, 130),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if (over_car and not self._prev_over_car) or (over_moto and not self._prev_over_moto):
                    winsound_beep()  # solo suena en Windows
                self._prev_over_car = over_car
                self._prev_over_moto = over_moto

                # 7.8) CSV por eventos (si está habilitado)
                self._write_event_rows(
                    car_in=car_in, car_out=car_out,
                    moto_in=moto_in, moto_out=moto_out,
                    car_inv=car_inv, moto_inv=moto_inv,
                )

                # 7.9) Mostrar / teclado (UI)
                if self.display:
                    cv2.imshow(WINDOW_NAME, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.stop_event.set()
                        break

        except Exception as e:
            self._notify_error(f"Fallo inesperado en el procesamiento: {e}")
        finally:
            # Liberación de recursos (siempre)
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

            # Notificar fin al frontend
            if self.on_finish:
                try:
                    self.on_finish()
                except Exception:
                    pass
