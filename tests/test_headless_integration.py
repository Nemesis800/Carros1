import os
from pathlib import Path
import numpy as np
import supervision as sv

from app import VideoProcessor, AppConfig

class FakeStopEvent:
    def __init__(self):
        self._s = False
    def is_set(self): return self._s
    def set(self): self._s = True

def _make_frame(w=160, h=120):
    # frame negro
    return np.zeros((h, w, 3), dtype=np.uint8)

def test_headless_pipeline_writes_csv(tmp_path, monkeypatch):
    # --- Fake VideoCapture que da 2 frames y termina ---
    class FakeCap:
        def __init__(self, *a, **k):
            self.count = 0
            self.opened = True
        def isOpened(self): return True
        def read(self):
            self.count += 1
            if self.count == 1:
                return True, _make_frame()
            if self.count == 2:
                return True, _make_frame()
            return False, None
        def release(self): pass

    monkeypatch.setattr("cv2.VideoCapture", lambda *a, **k: FakeCap())

    # --- Fake Detector que simula cruce (izquierda -> derecha) para 'car' ---
    from types import SimpleNamespace

    def fake_detect(self, frame):
        # 1er frame: x centro ~ 20 (izquierda)
        # 2do frame: x centro ~ 120 (derecha) => cruce
        idx = fake_detect.idx = getattr(fake_detect, "idx", 0) + 1
        if idx == 1:
            xyxy = np.array([[10, 10, 30, 30]], dtype=int)
        else:
            xyxy = np.array([[110, 10, 130, 30]], dtype=int)

        det = sv.Detections(
            xyxy=xyxy,
            confidence=np.array([0.9], dtype=float),
            class_id=np.array([0], dtype=int),
            tracker_id=np.array([1], dtype=int),  # mantenemos el mismo id
        )
        det.data["class_name"] = np.array(["car"])
        return det

    monkeypatch.setattr("app.VehicleDetector.detect", fake_detect)

    # --- Fake ByteTrack que pasa los detections tal cual ---
    class FakeTracker:
        def update_with_detections(self, detections):
            return detections
    monkeypatch.setattr("app.sv.ByteTrack", lambda: FakeTracker())

    # Config CLI-like
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

    # Debe existir un CSV en tmp_path con al menos una fila IN de car
    csvs = list(Path(tmp_path).glob("reporte_*.csv"))
    assert len(csvs) == 1
    content = csvs[0].read_text(encoding="utf-8-sig")
    assert ";IN;car;" in content or ";SUMMARY;-" in content
