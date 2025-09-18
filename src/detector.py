# src/detector.py
"""
Módulo de detección de vehículos con YOLO (Ultralytics).

Envuelve un modelo YOLO para detectar vehículos de interés
(actualmente: 'car' y 'motorcycle'), con soporte de fallback entre
distintos modelos (yolo12n.pt, yolo11n.pt, yolov8n.pt).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import supervision as sv
from ultralytics import YOLO


class VehicleDetector:
    """Detector de vehículos basado en YOLO.

    Permite detectar clases específicas ('car' y 'motorcycle'),
    normalizando nombres de salida y unificando IDs entre distintos
    modelos compatibles.

    Attributes:
        model: Modelo YOLO cargado.
        conf: Umbral de confianza.
        iou: Umbral de IoU.
        device: Dispositivo de ejecución (CPU/GPU).
        target_class_ids: IDs numéricos de clases de interés.
        id_to_unified_label: Mapeo id → nombre unificado ('car'/'motorcycle').
    """

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        conf: float = 0.3,
        iou: float = 0.5,
        device: str | int | None = None,
        classes_whitelist: List[str] | None = None,
    ) -> None:
        # -----------------------------
        # Estrategia de carga con fallbacks
        # -----------------------------
        preferred = [model_name]
        for alt in ["yolo12n.pt", "yolo11n.pt", "yolov8n.pt"]:
            if alt not in preferred:
                preferred.append(alt)

        self.model = None
        last_exc: Exception | None = None
        for cand in preferred:
            try:
                self.model = YOLO(cand)
                break
            except Exception as e:
                last_exc = e

        if self.model is None:
            raise RuntimeError(
                f"No se pudo cargar ningún modelo de la lista {preferred}. "
                f"Último error: {last_exc}"
            )

        # Umbrales de inferencia y dispositivo
        self.conf = conf
        self.iou = iou
        self.device = device

        # -----------------------------
        # Mapeo de ids ↔ nombres del modelo
        # -----------------------------
        try:
            model_names = self.model.names  # type: ignore[attr-defined]
        except Exception:
            model_names = getattr(self.model.model, "names", {})

        if isinstance(model_names, dict):
            id_to_name = {int(k): str(v) for k, v in model_names.items()}
        else:
            id_to_name = {i: str(n) for i, n in enumerate(list(model_names))}

        # -----------------------------
        # Selección de clases objetivo
        # -----------------------------
        target_labels = ["car", "motorcycle", "motorbike"]
        if classes_whitelist:
            target_labels = [c.lower() for c in classes_whitelist]

        # name_to_id: nombre unificado -> id del modelo
        self.name_to_id: Dict[str, int] = {}
        for cid, cname in id_to_name.items():
            cname_l = cname.lower()
            if cname_l in target_labels:
                if cname_l == "motorbike":
                    cname_l = "motorcycle"  # Normalización
                if cname_l not in self.name_to_id:
                    self.name_to_id[cname_l] = cid

        # Lista de ids a filtrar en inferencia
        self.target_class_ids: List[int] = list(self.name_to_id.values())

        # id_to_unified_label: id -> nombre unificado ('car'/'motorcycle')
        self.id_to_unified_label: Dict[int, str] = {
            cid: name for name, cid in self.name_to_id.items()
        }

        if not self.target_class_ids:
            raise RuntimeError(
                "No se encontraron clases objetivo ('car'/'motorcycle') "
                "en el modelo YOLO cargado."
            )

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """Corre inferencia sobre un frame y devuelve detecciones filtradas.

        Args:
            frame: Imagen en formato BGR (np.ndarray).

        Returns:
            Detecciones de Supervision (`sv.Detections`) filtradas a
            clases objetivo, con `.data['class_name']` agregado.
        """
        # Ejecutamos YOLO filtrando por clases objetivo
        results = self.model(
            frame,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
            classes=self.target_class_ids,
        )
        result = results[0]

        # Convertimos a detecciones de Supervision
        detections = sv.Detections.from_ultralytics(result)
        if len(detections) == 0:
            return detections

        # Filtro extra de seguridad
        mask = np.isin(detections.class_id, np.array(self.target_class_ids))
        detections = detections[mask]
        if len(detections) == 0:
            return detections

        # Agregar nombres de clase unificados ('car'/'motorcycle')
        class_names = np.array(
            [self.id_to_unified_label.get(int(cid), str(cid)) for cid in detections.class_id]
        )
        detections.data["class_name"] = class_names
        return detections


