# src/counter.py
"""
Módulo para conteo de cruces de vehículos en una línea definida.

Incluye:
- `_side_of_line`: utilidad para determinar en qué lado de una línea se encuentra un punto.
- `LineCrossingCounterByClass`: clase que lleva inventario de cruces IN/OUT por clase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Tuple

import numpy as np
import supervision as sv

# Alias legible para puntos (x, y) en píxeles
Point = Tuple[int, int]


def _side_of_line(point: Point, a: Point, b: Point) -> float:
    """Devuelve el signo de un punto respecto a una línea AB usando producto cruzado 2D.

    Interpretación del resultado:
        > 0 : punto a un lado de la línea
        < 0 : punto al lado opuesto
        = 0 : punto sobre la línea

    Esto permite detectar cambios de lado entre frames (cruces).
    """
    x, y = point
    x1, y1 = a
    x2, y2 = b
    return (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)


@dataclass
class LineCrossingCounterByClass:
    """Contador de cruces por clase con inventario dinámico.

    Args:
        a: Punto inicial de la línea de conteo.
        b: Punto final de la línea de conteo.
        labels: Clases a contabilizar (por defecto: "car", "motorcycle").
        invert_direction: Si es True, invierte el sentido IN/OUT.

    Atributos:
        in_counts: Conteo acumulado de entradas por clase.
        out_counts: Conteo acumulado de salidas por clase.
        inventory: Inventario actual por clase (IN - OUT).
        _last_side: Último lado visto para cada track_id.
    """

    # --- Parámetros de inicialización ---
    a: Point
    b: Point
    labels: Iterable[str] = ("car", "motorcycle")
    invert_direction: bool = False

    # --- Estado interno de conteo ---
    in_counts: Dict[str, int] = field(default_factory=dict)
    out_counts: Dict[str, int] = field(default_factory=dict)
    inventory: Dict[str, int] = field(default_factory=dict)

    # Trackeo interno: último lado visto por cada track_id
    _last_side: Dict[int, float] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        """Inicializa contadores para todas las etiquetas configuradas."""
        for label in self.labels:
            self.in_counts.setdefault(label, 0)
            self.out_counts.setdefault(label, 0)
            self.inventory.setdefault(label, 0)

    def reset(self) -> None:
        """Reinicia los contadores y limpia el estado de trackeo."""
        for label in self.labels:
            self.in_counts[label] = 0
            self.out_counts[label] = 0
            self.inventory[label] = 0
        self._last_side.clear()

    def update(self, detections: sv.Detections) -> None:
        """Actualiza el conteo con las detecciones de un frame.

        Requisitos de `detections`:
            - `.xyxy` (np.ndarray N×4): bounding boxes.
            - `.tracker_id` (np.ndarray N): IDs persistentes entre frames.
            - `detections.data["class_name"]`: nombres de clase normalizados.
        """
        if len(detections) == 0:
            # No hay detecciones en este frame
            # (Opcional: se podría implementar timeout para IDs "fantasma")
            return

        xyxy = detections.xyxy.astype(int)
        tracker_ids = detections.tracker_id
        class_names = detections.data.get("class_name", None)

        current_ids = set()

        for i in range(len(detections)):
            tid = tracker_ids[i] if tracker_ids is not None else None
            if tid is None:
                # Sin ID de tracker no se puede determinar cruce persistente
                continue
            current_ids.add(int(tid))

            # Centroide del bounding box
            x1, y1, x2, y2 = xyxy[i].tolist()
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Determinar lado actual respecto a la línea
            side = _side_of_line((cx, cy), self.a, self.b)

            # Lado previo registrado
            prev = self._last_side.get(int(tid))

            # Guardar último valor no nulo
            self._last_side[int(tid)] = side if side != 0 else (prev if prev is not None else 0)

            # Si no hay historial o punto está sobre la línea, no se cuenta
            if prev is None or prev == 0 or side == 0:
                continue

            # Cruce = cambio de signo entre prev y side
            crossed = (prev > 0 and side < 0) or (prev < 0 and side > 0)
            if not crossed:
                continue

            # Clase del objeto
            cname = str(class_names[i]) if class_names is not None else "unknown"

            # Determinar dirección del cruce
            went_neg_to_pos = prev < 0 and side > 0
            if self.invert_direction:
                went_neg_to_pos = not went_neg_to_pos

            # Actualizar contadores
            if went_neg_to_pos:
                self.in_counts[cname] = self.in_counts.get(cname, 0) + 1
                self.inventory[cname] = self.inventory.get(cname, 0) + 1
            else:
                self.out_counts[cname] = self.out_counts.get(cname, 0) + 1
                self.inventory[cname] = self.inventory.get(cname, 0) - 1

        # Limpiar IDs no presentes en este frame
        stale_ids = [k for k in self._last_side if k not in current_ids]
        for k in stale_ids:
            del self._last_side[k]

