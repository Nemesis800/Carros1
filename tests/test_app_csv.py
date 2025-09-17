import io
import csv
from types import SimpleNamespace

from app import VideoProcessor, AppConfig

class _FakeStopEvent:
    def __init__(self):
        self._s = False
    def is_set(self): return self._s
    def set(self): self._s = True

def test_write_event_rows_and_summary(tmp_reports_dir, monkeypatch):
    # Config mínima
    cfg = AppConfig(
        model_name="yolo12n.pt",
        conf=0.30,
        line_orientation="vertical",
        line_position=0.5,
        invert_direction=False,
        capacity_car=50, capacity_moto=50,
        enable_csv=True, csv_dir=str(tmp_reports_dir),
    )

    vp = VideoProcessor(video_source="video.mp4", config=cfg,
                        stop_event=_FakeStopEvent(),
                        on_error=None, on_finish=None)

    # Redirigimos a un buffer en memoria y usamos ';' como en tu app
    buf = io.StringIO()
    vp.csv_writer = csv.writer(buf, delimiter=';')

    # Estado previo (todo en cero)
    assert vp._prev_counts == {"car_in":0,"car_out":0,"moto_in":0,"moto_out":0}

    # Simulamos que el inventario terminó en:
    vp._last_car_inv = 2
    vp._last_moto_inv = 0

    # 1) Escribimos eventos: entró 1 carro, y salió 1 moto (ejemplo)
    vp._write_event_rows(
        car_in=1, car_out=0,
        moto_in=0, moto_out=1,
        car_inv=1, moto_inv=-1,
    )
    # Debe haber 2 filas (un IN car, un OUT moto)
    lines = [l for l in buf.getvalue().strip().splitlines() if l]
    assert len(lines) == 2
    assert ";IN;car;" in lines[0] or ";IN;car;" in lines[1]
    assert ";OUT;motorcycle;" in lines[0] or ";OUT;motorcycle;" in lines[1]

    # 2) SUMMARY
    before_summary_lines = len(lines)
    vp._write_summary()
    lines = [l for l in buf.getvalue().strip().splitlines() if l]
    assert len(lines) == before_summary_lines + 1
    assert ";SUMMARY;-" in lines[-1]
