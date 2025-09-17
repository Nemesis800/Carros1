from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, Iterable
import numpy as np
import supervision as sv

# Alias legible para puntos (x, y) en píxeles
Point = Tuple[int, int]


def _side_of_line(point: Point, a: Point, b: Point) -> float:
    """
    Devuelve el signo del punto respecto a la línea AB usando el producto cruzado 2D.
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
    """
    Contador de cruces por clase con inventario.

    Parámetros:
      - a, b: puntos que definen la línea de conteo.
      - labels: clases que se contabilizan (por defecto, 'car' y 'motorcycle').
      - invert_direction: si es True, invierte el sentido IN/OUT.

    Estado:
      - in_counts / out_counts: acumuladores por clase.
      - inventory: IN - OUT por clase (inventario actual).
      - _last_side: último signo observado por track_id (para detectar cambios).
    """
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
        """Inicializa contadores para todas las etiquetas configuradas."""
        for l in self.labels:
            self.in_counts.setdefault(l, 0)
            self.out_counts.setdefault(l, 0)
            self.inventory.setdefault(l, 0)

    def reset(self) -> None:
        """Reinicia los contadores y limpia el estado de trackeo."""
        for l in self.labels:
            self.in_counts[l] = 0
            self.out_counts[l] = 0
            self.inventory[l] = 0
        self._last_side.clear()

    def update(self, detections: sv.Detections) -> None:
        """
        Actualiza el conteo con las detecciones del frame actual.
        Requisitos de 'detections':
          - .xyxy (np.ndarray N×4)
          - .tracker_id (np.ndarray N) con IDs persistentes entre frames
          - detections.data["class_name"] con nombres de clase normalizados
        """
        if len(detections) == 0:
            # No hay detecciones en este frame: nada que hacer
            # (Opcional: aquí se podría implementar timeout de IDs “fantasma”)
            return

        xyxy = detections.xyxy.astype(int)
        tracker_ids = detections.tracker_id
        class_names = detections.data.get("class_name", None)

        current_ids = set()
        for i in range(len(detections)):
            tid = tracker_ids[i] if tracker_ids is not None else None
            if tid is None:
                # Sin ID de tracker no podemos determinar cruce persistente
                continue
            current_ids.add(int(tid))

            # Centro del bounding box
            x1, y1, x2, y2 = xyxy[i].tolist()
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Lado actual respecto a la línea AB
            side = _side_of_line((cx, cy), self.a, self.b)

            # Lado previo registrado para este track_id
            prev = self._last_side.get(int(tid))

            # Actualizamos el lado guardando el último valor no nulo
            self._last_side[int(tid)] = side if side != 0 else (prev if prev is not None else 0)

            # Si no hay historial o uno de los lados es 0 (sobre la línea), no contamos cruce
            if prev is None or prev == 0 or side == 0:
                continue

            # Se considera cruce si hay cambio de signo (de + a − o de − a +)
            crossed = (prev > 0 and side < 0) or (prev < 0 and side > 0)
            if not crossed:
                continue

            # Nombre de clase (normalizado por el detector a 'car'/'motorcycle')
            cname = str(class_names[i]) if class_names is not None else "unknown"

            # Dirección del cruce: prev<0 -> side>0 se interpreta como “negativo a positivo”
            went_neg_to_pos = prev < 0 and side > 0
            if self.invert_direction:
                # Invertimos la semántica del sentido IN/OUT
                went_neg_to_pos = not went_neg_to_pos

            # Actualizamos contadores e inventario
            if went_neg_to_pos:
                self.in_counts[cname] = self.in_counts.get(cname, 0) + 1
                self.inventory[cname] = self.inventory.get(cname, 0) + 1
            else:
                self.out_counts[cname] = self.out_counts.get(cname, 0) + 1
                self.inventory[cname] = self.inventory.get(cname, 0) - 1

        # Purgamos IDs que ya no están presentes en este frame para no crecer sin límite
        stale_ids = [k for k in self._last_side.keys() if k not in current_ids]
        for k in stale_ids:
            del self._last_side[k]
