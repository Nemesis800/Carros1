# tests/test_counter.py
"""
Tests para el módulo `counter.py`.

Escenarios validados:
1. `_side_of_line` devuelve signos opuestos para lados distintos y 0 sobre la línea.
2. Conteo IN cuando un objeto cruza de lado negativo a positivo.
3. Conteo OUT cuando la dirección está invertida.
4. Purga de IDs que ya no aparecen en el frame actual.
"""

import numpy as np
import supervision as sv

from counter import LineCrossingCounterByClass, _side_of_line


def _detections_from_xyxy(ids, xyxy, class_ids, class_names) -> sv.Detections:
    """Helper para crear `sv.Detections` simulados con tracker_id y class_name."""
    det = sv.Detections(
        xyxy=np.array(xyxy, dtype=int),
        confidence=np.array([0.9] * len(xyxy), dtype=float),
        class_id=np.array(class_ids, dtype=int),
        tracker_id=np.array(ids, dtype=int),
    )
    det.data["class_name"] = np.array(class_names)
    return det


def test_side_of_line_sign():
    """Verifica que `_side_of_line` distingue lados y punto sobre línea."""
    a, b = (0, 0), (10, 0)  # Línea horizontal en y=0
    s_up = _side_of_line((5, 1), a, b)  # Punto arriba
    s_down = _side_of_line((5, -1), a, b)  # Punto abajo
    s_on = _side_of_line((5, 0), a, b)  # Punto sobre la línea

    assert s_up * s_down < 0  # lados opuestos => signos opuestos
    assert s_on == 0  # sobre la línea => 0


def test_count_in_when_crossing_neg_to_pos():
    """Un auto cruza de izquierda a derecha y se cuenta como IN."""
    a, b = (50, 0), (50, 100)  # Línea vertical en x=50
    counter = LineCrossingCounterByClass(a=a, b=b, invert_direction=False)

    # Frame 1: auto a la izquierda (centro < 50)
    det1 = _detections_from_xyxy([1], [(10, 10, 20, 20)], [0], ["car"])
    counter.update(det1)

    # Frame 2: auto cruza a la derecha (centro > 50)
    det2 = _detections_from_xyxy([1], [(80, 10, 90, 20)], [0], ["car"])
    counter.update(det2)

    assert counter.in_counts.get("car", 0) == 1
    assert counter.out_counts.get("car", 0) == 0
    assert counter.inventory.get("car", 0) == 1


def test_count_out_when_inverted_direction():
    """Con invert_direction=True, el cruce neg→pos se cuenta como OUT."""
    a, b = (50, 0), (50, 100)
    counter = LineCrossingCounterByClass(a=a, b=b, invert_direction=True)

    det1 = _detections_from_xyxy([7], [(10, 10, 20, 20)], [0], ["motorcycle"])
    counter.update(det1)
    det2 = _detections_from_xyxy([7], [(80, 10, 90, 20)], [0], ["motorcycle"])
    counter.update(det2)

    assert counter.in_counts.get("motorcycle", 0) == 0
    assert counter.out_counts.get("motorcycle", 0) == 1
    assert counter.inventory.get("motorcycle", 0) == -1


def test_purge_stale_ids():
    """Verifica que IDs desaparecidos en un frame se purgan del estado interno."""
    a, b = (0, 50), (100, 50)
    counter = LineCrossingCounterByClass(a=a, b=b)

    # Frame 1: aparece id=1
    det1 = _detections_from_xyxy([1], [(10, 10, 20, 20)], [0], ["car"])
    counter.update(det1)
    assert 1 in counter._last_side

    # Frame 2: id=1 desaparece, aparece id=2
    det2 = _detections_from_xyxy([2], [(30, 10, 40, 20)], [0], ["car"])
    counter.update(det2)

    assert 1 not in counter._last_side
    assert 2 in counter._last_side

