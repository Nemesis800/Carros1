from detector import VehicleDetector

class FakeYOLO:
    """Simula un modelo Ultralytics con .names (dict o list)."""
    def __init__(self, *args, **kwargs):
        # ids -> nombres (incluye 'motorbike' para probar normalización)
        self.names = {0: "person", 1: "car", 2: "motorbike", 3: "dog"}

def test_mapping_and_unification(monkeypatch):
    # Monkeypatch: cuando se llame ultralytics.YOLO, devolvemos FakeYOLO
    import detector as mod
    monkeypatch.setattr(mod, "YOLO", lambda *a, **k: FakeYOLO())

    vd = VehicleDetector(model_name="yolo12n.pt", conf=0.3, iou=0.5)
    # Debe contener car y motorcycle (unificando motorbike->motorcycle)
    assert set(vd.name_to_id.keys()) == {"car", "motorcycle"}

    # Los ids apuntan a los del fake
    assert vd.id_to_unified_label[1] == "car"
    assert vd.id_to_unified_label[2] == "motorcycle"
