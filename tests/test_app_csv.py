# tests/test_app_csv.py
"""
Tests para validación de escritura de eventos y resumen en CSV
usando la clase `VideoProcessor`.

Escenarios probados:
1. Escritura de filas de eventos (IN/OUT por clase).
2. Escritura de fila de resumen (SUMMARY).
"""

import csv
import io

from app import AppConfig, VideoProcessor


class _FakeStopEvent:
    """Implementación mínima de threading.Event para pruebas."""

    def __init__(self) -> None:
        self._s = False

    def is_set(self) -> bool:
        return self._s

    def set(self) -> None:
        self._s = True


def test_write_event_rows_and_summary(tmp_reports_dir, monkeypatch) -> None:
    """Verifica que VideoProcessor escriba correctamente eventos y resumen en CSV."""
    # Configuración mínima
    cfg = AppConfig(
        model_name="yolo12n.pt",
        conf=0.30,
        line_orientation="vertical",
        line_position=0.5,
        invert_direction=False,
        capacity_car=50,
        capacity_moto=50,
        enable_csv=True,
        csv_dir=str(tmp_reports_dir),
    )

    vp = VideoProcessor(
        video_source="video.mp4",
        config=cfg,
        stop_event=_FakeStopEvent(),
        on_error=None,
        on_finish=None,
    )

    # Usamos un buffer en memoria en lugar de archivo real
    buf = io.StringIO()
    vp.csv_writer = csv.writer(buf, delimiter=";")

    # Estado inicial esperado
    assert vp._prev_counts == {"car_in": 0, "car_out": 0, "moto_in": 0, "moto_out": 0}

    # Simulamos inventario final
    vp._last_car_inv = 2
    vp._last_moto_inv = 0

    # 1) Eventos simulados: entra 1 carro, sale 1 moto
    vp._write_event_rows(
        car_in=1,
        car_out=0,
        moto_in=0,
        moto_out=1,
        car_inv=1,
        moto_inv=-1,
    )

    # Debe escribir 2 filas (un IN car, un OUT motorcycle)
    lines = [l for l in buf.getvalue().strip().splitlines() if l]
    assert len(lines) == 2
    assert any(";IN;car;" in l for l in lines)
    assert any(";OUT;motorcycle;" in l for l in lines)

    # 2) SUMMARY
    before_summary = len(lines)
    vp._write_summary()
    lines = [l for l in buf.getvalue().strip().splitlines() if l]

    # Debe haberse agregado una fila más con "SUMMARY"
    assert len(lines) == before_summary + 1
    assert ";SUMMARY;-" in lines[-1]

