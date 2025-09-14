from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, Iterable
import numpy as np
import supervision as sv

Point = Tuple[int, int]


def _side_of_line(point: Point, a: Point, b: Point) -> float:
    """Devuelve el signo del punto respecto a la línea AB usando el producto cruzado.
    > 0: un lado, < 0: lado opuesto, 0: sobre la línea.
    """
    x, y = point
    x1, y1 = a
    x2, y2 = b
    return (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)


@dataclass
class LineCrossingCounterByClass:
    a: Point
    b: Point
    labels: Iterable[str] = ("car", "motorcycle")
    invert_direction: bool = False  # True: invierte sentido 'in'/'out'

    in_counts: Dict[str, int] = field(default_factory=dict)
    out_counts: Dict[str, int] = field(default_factory=dict)
    inventory: Dict[str, int] = field(default_factory=dict)

    # Estado interno: último lado visto por cada track_id
    _last_side: Dict[int, float] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        for l in self.labels:
            self.in_counts.setdefault(l, 0)
            self.out_counts.setdefault(l, 0)
            self.inventory.setdefault(l, 0)

    def reset(self) -> None:
        for l in self.labels:
            self.in_counts[l] = 0
            self.out_counts[l] = 0
            self.inventory[l] = 0
        self._last_side.clear()

    def update(self, detections: sv.Detections) -> None:
        if len(detections) == 0:
            # Limpieza suave: elimina IDs no vistos por un tiempo podría ser agregado
            return

        xyxy = detections.xyxy.astype(int)
        tracker_ids = detections.tracker_id
        class_names = detections.data.get("class_name", None)

        current_ids = set()
        for i in range(len(detections)):
            tid = tracker_ids[i] if tracker_ids is not None else None
            if tid is None:
                continue
            current_ids.add(int(tid))

            x1, y1, x2, y2 = xyxy[i].tolist()
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            side = _side_of_line((cx, cy), self.a, self.b)

            prev = self._last_side.get(int(tid))
            self._last_side[int(tid)] = side if side != 0 else prev if prev is not None else 0

            if prev is None or prev == 0 or side == 0:
                # Aún no hay información clara de cruce
                continue

            crossed = (prev > 0 and side < 0) or (prev < 0 and side > 0)
            if not crossed:
                continue

            cname = str(class_names[i]) if class_names is not None else "unknown"
            # Determinar dirección del cruce
            went_neg_to_pos = prev < 0 and side > 0
            if self.invert_direction:
                went_neg_to_pos = not went_neg_to_pos

            if went_neg_to_pos:
                self.in_counts[cname] = self.in_counts.get(cname, 0) + 1
                self.inventory[cname] = self.inventory.get(cname, 0) + 1
            else:
                self.out_counts[cname] = self.out_counts.get(cname, 0) + 1
                self.inventory[cname] = self.inventory.get(cname, 0) - 1

        # Opción: purgar IDs no presentes para evitar crecimiento indefinido
        stale_ids = [k for k in self._last_side.keys() if k not in current_ids]
        for k in stale_ids:
            del self._last_side[k]
