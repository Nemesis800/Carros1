# tests/test_headless_integration.py
"""
Prueba de integración (headless) del pipeline:

- Silencia MLflow para que no sea requisito en el entorno de test.
- Mockea `cv2.VideoCapture` para producir 2 frames sintéticos.
- Mockea el detector para simular un cruce izquierda→derecha de un 'car'.
- Mockea ByteTrack para retornar las detecciones sin cambios.
- Ejecuta `VideoProcessor.run()` en modo headless y verifica que se genere
  al menos un archivo CSV con contenido esperado.
"""

from __future__ import annotations

import sys
from pathlib import Path
import types

import cv2
import numpy as np
import supervision as sv


class FakeStopEvent:
    """Implementación mínima de Event para el test (no bloqueante)."""

    def __init__(self) -> None:
        self._s = False

    def is_set(self) -> bool:
        return self._s

    def set(self) -> None:
        self._s = True


def _make_frame(w: int = 160, h: int = 120) -> np.ndarray:
    """Crea un frame negro sintético (BGR)."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_headless_pipeline_writes_csv(tmp_path, monkeypatch) -> None:
    """Verifica que el pipeline headless genere un CSV con contenido válido."""
    # --- 1) Silenciar MLflow en todo el pipeline importado ---
    noop = types.SimpleNamespace()

    def _retself(*_args, **_kwargs):
        return noop

    # Cualquier atributo o llamada retorna el propio `noop`
    noop.__getattr__ = lambda *_a, **_k: noop
    noop.__call__ = _retself
    monkeypatch.setitem(sys.modules, "mlflow", noop)

    # --- 2) Fake VideoCapture: 2 frames + .get() para frame count ---
    class FakeCap:
        def __init__(self, *_a, **_k):
            self.count = 0

        def isOpened(self) -> bool:
            return True

        def read(self):
            self.count += 1
            if self.count == 1:
                return True, _make_frame()
            if self.count == 2:
                return True, _make_frame()
            return False, None

        def get(self, prop):
            return 2 if prop == cv2.CAP_PROP_FRAME_COUNT else 0

        def release(self) -> None:
            pass

    monkeypatch.setattr("cv2.VideoCapture", lambda *_a, **_k: FakeCap())

    # --- 3) Fake Detector: simula cruce izq->der para 'car' ---
    def fake_detect(self, frame):
        idx = fake_detect.idx = getattr(fake_detect, "idx", 0) + 1
        if idx == 1:
            xyxy = np.array([[10, 10, 30, 30]], dtype=int)   # centro x < 50
        else:
            xyxy = np.array([[110, 10, 130, 30]], dtype=int)  # centro x > 50
        det = sv.Detections(
            xyxy=xyxy,
            confidence=np.array([0.9], dtype=float),
            class_id=np.array([0], dtype=int),
            tracker_id=np.array([1], dtype=int),
        )
        det.data["class_name"] = np.array(["car"])
        return det

    monkeypatch.setattr("detector.VehicleDetector.detect", fake_detect)

    # --- 4) Fake ByteTrack: pasa detections tal cual ---
    class FakeTracker:
        def update_with_detections(self, detections):
            return detections

    monkeypatch.setattr("processor.sv.ByteTrack", lambda: FakeTracker())

    # --- 5) Importar y ejecutar pipeline en modo headless ---
    from config import AppConfig
    from processor import VideoProcessor

    cfg = AppConfig(
        model_name="yolo11n.pt",
        conf=0.3,
        line_orientation="vertical",
        line_position=0.5,
        invert_direction=False,
        capacity_car=5,
        capacity_moto=5,
        enable_csv=True,
        csv_dir=str(tmp_path),
        csv_name="Registro",  # nombre estable para el test
    )

    vp = VideoProcessor(
        video_source="fake.mp4",
        config=cfg,
        stop_event=FakeStopEvent(),
        on_error=None,
        on_finish=None,
        display=False,  # headless
    )
    vp.run()

    # --- 6) Verificar que se generó un CSV con algo de contenido esperado ---
    csvs = list(Path(tmp_path).glob("*.csv"))
    assert csvs, "No se generó ningún CSV en el directorio temporal"

    content = csvs[0].read_text(encoding="utf-8-sig")
    assert ";IN;car;" in content or ";SUMMARY;-" in content, "CSV sin eventos ni summary"
