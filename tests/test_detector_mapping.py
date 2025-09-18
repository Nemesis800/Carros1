# tests/test_detector_mapping.py
"""
Tests para `detector.VehicleDetector` enfocados en el mapeo de clases.

Se valida que:
- Se unifique correctamente "motorbike" → "motorcycle".
- Los diccionarios internos (`name_to_id`, `id_to_unified_label`)
  contengan las clases de interés y apunten a los IDs esperados.
"""

from detector import VehicleDetector


class FakeYOLO:
    """Simula un modelo Ultralytics con atributo `.names` (dict)."""

    def __init__(self, *args, **kwargs) -> None:
        # ids → nombres (incluye 'motorbike' para probar normalización)
        self.names = {0: "person", 1: "car", 2: "motorbike", 3: "dog"}


def test_mapping_and_unification(monkeypatch):
    """Verifica que 'motorbike' se unifica como 'motorcycle' en VehicleDetector."""
    import detector as mod

    # Sustituimos YOLO por FakeYOLO
    monkeypatch.setattr(mod, "YOLO", lambda *a, **k: FakeYOLO())

    vd = VehicleDetector(model_name="yolo12n.pt", conf=0.3, iou=0.5)

    # Debe contener car y motorcycle como claves unificadas
    assert set(vd.name_to_id.keys()) == {"car", "motorcycle"}

    # Los IDs deben apuntar a los correctos del fake
    assert vd.id_to_unified_label[1] == "car"
    assert vd.id_to_unified_label[2] == "motorcycle"
