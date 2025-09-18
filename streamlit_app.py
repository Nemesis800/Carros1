# streamlit_app.py
"""
Aplicación Streamlit para detección y conteo de vehículos.

Características:
- Configuración de fuente de video (archivo o webcam).
- Selección de modelo YOLO y umbral de confianza.
- Definición de línea de conteo (posición, orientación, dirección).
- Configuración de capacidades y guardado CSV.
- Visualización en tiempo real de frames, progreso y errores.
"""

from __future__ import annotations

import os
import sys
import time
import threading
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue

import numpy as np
import streamlit as st

# --- Asegurar imports desde src/ ---
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import AppConfig  # noqa: E402
from processor import VideoProcessor  # noqa: E402


# ----------------------------------------------------------------------
# Estado de sesión
# ----------------------------------------------------------------------
def _init_session() -> None:
    """Inicializa variables de session_state usadas en la app."""
    ss = st.session_state
    ss.setdefault("thread", None)
    ss.setdefault("stop_event", None)
    ss.setdefault("running", False)

    # Estado visible
    ss.setdefault("last_csv", None)
    ss.setdefault("last_error", None)
    ss.setdefault("last_frame", None)
    ss.setdefault("progress", 0.0)

    # Colas de comunicación (solo se crean una vez)
    if "frame_q" not in ss:
        ss.frame_q = Queue(maxsize=1)
    if "progress_q" not in ss:
        ss.progress_q = Queue(maxsize=8)
    if "finish_q" not in ss:
        ss.finish_q = Queue(maxsize=1)
    if "error_q" not in ss:
        ss.error_q = Queue(maxsize=8)


# ----------------------------------------------------------------------
# Configuración de página
# ----------------------------------------------------------------------
_init_session()
st.set_page_config(page_title="Conteo de Vehículos", layout="centered")
st.title("🚗 Detección y Conteo de Vehículos (Streamlit)")

# ----------------------------------------------------------------------
# Sidebar con parámetros de ejecución
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("Configuración")

    # Fuente
    use_webcam = st.toggle("Usar webcam (ID 0)", value=False)
    uploaded = None
    if not use_webcam:
        uploaded = st.file_uploader(
            "Sube un video (mp4/avi/mov/mkv)",
            type=["mp4", "avi", "mov", "mkv"],
        )

    # Modelo y confianza
    model = st.selectbox("Modelo YOLO", ["yolo11n.pt", "yolov8n.pt", "yolo12n.pt"], index=0)
    conf = st.slider("Confianza", 0.10, 0.80, 0.30, step=0.01)

    # Línea de conteo
    orientation = st.selectbox("Orientación de línea", ["horizontal", "vertical"], index=1)
    line_pos = st.slider("Posición de la línea", 0.10, 0.90, 0.50, step=0.01)
    invert_dir = st.toggle("Invertir dirección (IN ↔ OUT)", value=False)

    # Capacidades
    cap_car = st.number_input("Capacidad carros", min_value=0, value=50, step=1)
    cap_moto = st.number_input("Capacidad motos", min_value=0, value=50, step=1)

    # CSV
    enable_csv = st.toggle("Guardar CSV de eventos", value=True)
    csv_dir = st.text_input("Carpeta CSV", value=str(ROOT / "reports"))
    csv_name = st.text_input("Nombre de archivo (sin .csv)", value="")

    st.divider()
    run_btn = st.button("▶️ Procesar")
    stop_btn = st.button("⏹️ Detener")


# ----------------------------------------------------------------------
# Funciones auxiliares
# ----------------------------------------------------------------------
def _save_uploaded_to_disk(file) -> str | None:
    """Guarda archivo subido en disco y devuelve la ruta."""
    if file is None:
        return None
    uploads = ROOT / "uploads"
    uploads.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = uploads / f"{ts}_{file.name}"
    with open(dst, "wb") as f:
        f.write(file.read())
    return str(dst)


# ----------------------------------------------------------------------
# Botón detener
# ----------------------------------------------------------------------
if stop_btn and st.session_state.running and st.session_state.stop_event:
    st.session_state.stop_event.set()

# ----------------------------------------------------------------------
# Botón procesar
# ----------------------------------------------------------------------
if run_btn and not st.session_state.running:
    # Resetear estado visible
    st.session_state.last_error = None
    st.session_state.last_csv = None
    st.session_state.last_frame = None
    st.session_state.progress = 0.0

    # Vaciar colas
    for qname in ("frame_q", "progress_q", "finish_q", "error_q"):
        q: Queue = getattr(st.session_state, qname)
        try:
            while True:
                q.get_nowait()
        except Empty:
            pass

    # Fuente
    if use_webcam:
        source: int | str = 0
    else:
        path = _save_uploaded_to_disk(uploaded)
        if not path or not os.path.exists(path):
            st.error("Debes subir un video válido o activar webcam.")
            st.stop()
        source = path

    # Configuración
    cfg = AppConfig(
        model_name=model,
        conf=float(conf),
        iou=0.5,
        device=None,
        line_orientation=orientation,
        line_position=float(line_pos),
        invert_direction=bool(invert_dir),
        capacity_car=int(cap_car),
        capacity_moto=int(cap_moto),
        enable_csv=bool(enable_csv),
        csv_dir=csv_dir.strip() or "reports",
        csv_name=csv_name.strip(),
    )

    stop_event = threading.Event()

    # Callbacks (no deben usar st.*)
    frame_q, progress_q, finish_q, error_q = (
        st.session_state.frame_q,
        st.session_state.progress_q,
        st.session_state.finish_q,
        st.session_state.error_q,
    )

    def cb_on_frame(frame_rgb: np.ndarray):
        try:
            if frame_q.full():
                frame_q.get_nowait()
            frame_q.put_nowait(frame_rgb)
        except Exception:
            pass

    def cb_on_progress(p: float):
        try:
            if progress_q.full():
                progress_q.get_nowait()
            progress_q.put_nowait(float(p))
        except Exception:
            pass

    def cb_on_error(msg: str):
        try:
            error_q.put_nowait(str(msg))
        except Exception:
            pass

    def make_cb_on_finish(vp: VideoProcessor):
        def _cb():
            info = {"csv": getattr(vp, "_csv_path_str", None)}
            if not finish_q.empty():
                finish_q.get_nowait()
            finish_q.put_nowait(info)
        return _cb

    # Instanciar procesador
    vp = VideoProcessor(
        video_source=source,
        config=cfg,
        stop_event=stop_event,
        on_error=cb_on_error,
        on_finish=None,
        display=False,
        on_frame=cb_on_frame,
        on_progress=cb_on_progress,
    )
    vp.on_finish = make_cb_on_finish(vp)

    st.session_state.thread = vp
    st.session_state.stop_event = stop_event
    st.session_state.running = True
    vp.start()


# ----------------------------------------------------------------------
# Visualización en UI
# ----------------------------------------------------------------------
frame_placeholder = st.empty()
progress_placeholder = st.progress(int(st.session_state.progress * 100))

# Frames
last = None
try:
    while True:
        last = st.session_state.frame_q.get_nowait()
except Empty:
    pass
if last is not None:
    st.session_state.last_frame = last

if st.session_state.last_frame is not None:
    frame_placeholder.image(st.session_state.last_frame, channels="RGB")

# Progreso
try:
    while True:
        p = st.session_state.progress_q.get_nowait()
        st.session_state.progress = float(p)
except Empty:
    pass
progress_placeholder.progress(int(st.session_state.progress * 100))

# Errores
try:
    while True:
        err = st.session_state.error_q.get_nowait()
        st.session_state.last_error = str(err)
except Empty:
    pass
if st.session_state.last_error:
    st.error(f"Error: {st.session_state.last_error}")

# Finalización
try:
    info = st.session_state.finish_q.get_nowait()
    st.session_state.running = False
    st.session_state.last_csv = info.get("csv")
except Empty:
    pass

if st.session_state.running:
    st.info("Procesando… puedes detener con el botón de la izquierda.")
else:
    if st.session_state.thread is None:
        st.success("Listo para procesar.")
    else:
        st.info("Ejecución finalizada.")

# CSV descarga
if st.session_state.last_csv:
    csv_path = st.session_state.last_csv
    if csv_path and os.path.exists(csv_path):
        st.success(f"CSV generado: {csv_path}")
        with open(csv_path, "rb") as f:
            st.download_button(
                "Descargar CSV",
                data=f.read(),
                file_name=Path(csv_path).name,
                mime="text/csv",
            )

# Auto-refresh
if st.session_state.running:
    time.sleep(0.5)
    try:
        st.experimental_rerun()
    except Exception:
        try:
            st.rerun()
        except Exception:
            pass

st.caption("Tip: usa videos cortos para probar. Si usas webcam, cierra otras apps que estén usando la cámara.")
