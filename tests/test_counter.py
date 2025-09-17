import numpy as np
import supervision as sv

from counter import LineCrossingCounterByClass, _side_of_line

def _detections_from_xyxy(ids, xyxy, class_ids, class_names):
    det = sv.Detections(
        xyxy=np.array(xyxy, dtype=int),
        confidence=np.array([0.9]*len(xyxy), dtype=float),
        class_id=np.array(class_ids, dtype=int),
        tracker_id=np.array(ids, dtype=int),
    )
    det.data["class_name"] = np.array(class_names)
    return det

def test_side_of_line_sign():
    a, b = (0, 0), (10, 0)  # línea horizontal (y=0)
    s_up = _side_of_line((5, 1), a, b)    # punto "arriba"
    s_down = _side_of_line((5, -1), a, b) # punto "abajo"
    s_on = _side_of_line((5, 0), a, b)    # sobre la línea

    # Lo importante: lados opuestos => signos opuestos, y el punto sobre la línea => 0
    assert s_up * s_down < 0
    assert s_on == 0

def test_count_in_when_crossing_neg_to_pos():
    # Línea vertical x=50
    a, b = (50, 0), (50, 100)
    counter = LineCrossingCounterByClass(a=a, b=b, invert_direction=False)

    # Frame 1: auto a la izquierda (x centro < 50)
    det1 = _detections_from_xyxy(
        ids=[1],
        xyxy=[(10, 10, 20, 20)],  # centro ~ (15, 15) -> x < 50
        class_ids=[0],
        class_names=["car"],
    )
    counter.update(det1)

    # Frame 2: auto cruza a la derecha (x centro > 50)
    det2 = _detections_from_xyxy(
        ids=[1],
        xyxy=[(80, 10, 90, 20)],  # centro ~ (85, 15) -> x > 50
        class_ids=[0],
        class_names=["car"],
    )
    counter.update(det2)

    assert counter.in_counts.get("car", 0) == 1
    assert counter.out_counts.get("car", 0) == 0
    assert counter.inventory.get("car", 0) == 1

def test_count_out_when_inverted_direction():
    # Misma escena pero invertimos IN/OUT
    a, b = (50, 0), (50, 100)
    counter = LineCrossingCounterByClass(a=a, b=b, invert_direction=True)

    det1 = _detections_from_xyxy([7], [(10, 10, 20, 20)], [0], ["motorcycle"])
    counter.update(det1)
    det2 = _detections_from_xyxy([7], [(80, 10, 90, 20)], [0], ["motorcycle"])
    counter.update(det2)

    # Con invert_direction=True, el cruce neg->pos debe contarse como OUT
    assert counter.in_counts.get("motorcycle", 0) == 0
    assert counter.out_counts.get("motorcycle", 0) == 1
    assert counter.inventory.get("motorcycle", 0) == -1

def test_purge_stale_ids():
    a, b = (0, 50), (100, 50)
    c = LineCrossingCounterByClass(a=a, b=b)

    # Frame 1: aparece id=1
    det1 = _detections_from_xyxy(
        ids=[1],
        xyxy=[(10, 10, 20, 20)],
        class_ids=[0],
        class_names=["car"],
    )
    c.update(det1)
    assert 1 in c._last_side  # quedó registrado

    # Frame 2: solo aparece id=2 (id=1 ya no está -> debe purgar id=1)
    det2 = _detections_from_xyxy(
        ids=[2],
        xyxy=[(30, 10, 40, 20)],
        class_ids=[0],
        class_names=["car"],
    )
    c.update(det2)

    assert 1 not in c._last_side
    assert 2 in c._last_side
