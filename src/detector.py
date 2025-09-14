from __future__ import annotations
import numpy as np
import supervision as sv
from ultralytics import YOLO
from typing import Dict, List


class VehicleDetector:
    """
    Envuelve un modelo YOLO de Ultralytics para detectar vehículos (car y motorcycle).
    Usa el modelo ligero por defecto: yolo11n.pt
    """

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        conf: float = 0.3,
        iou: float = 0.5,
        device: str | int | None = None,
        classes_whitelist: List[str] | None = None,
    ) -> None:
        # Cargar modelo con fallback si el nombre no está soportado en la versión instalada
        try:
            self.model = YOLO(model_name)
        except Exception as e:
            print(f"[WARN] No se pudo cargar '{model_name}': {e}. Probando 'yolov8n.pt' como fallback...")
            self.model = YOLO("yolov8n.pt")
        self.conf = conf
        self.iou = iou
        self.device = device

        # Obtener nombres de clases del modelo (manejar distintas versiones)
        try:
            model_names = self.model.names  # type: ignore[attr-defined]
        except Exception:
            model_names = getattr(self.model.model, "names", {})  # fallback

        # Normalizar a dict {id: name}
        if isinstance(model_names, dict):
            id_to_name = {int(k): str(v) for k, v in model_names.items()}
        else:
            id_to_name = {i: str(n) for i, n in enumerate(list(model_names))}

        # Por defecto, filtrar a car y motorcycle (aceptar 'motorbike')
        target_labels = ["car", "motorcycle", "motorbike"]
        if classes_whitelist:
            target_labels = [c.lower() for c in classes_whitelist]

        self.name_to_id: Dict[str, int] = {}
        for cid, cname in id_to_name.items():
            cname_l = cname.lower()
            if cname_l in target_labels:
                # Unificar 'motorbike' como 'motorcycle'
                if cname_l == "motorbike":
                    cname_l = "motorcycle"
                if cname_l not in self.name_to_id:
                    self.name_to_id[cname_l] = cid

        # Asegurar que existan al menos car y motorcycle si están en el modelo
        self.target_class_ids: List[int] = list(self.name_to_id.values())
        self.id_to_unified_label: Dict[int, str] = {
            cid: name for name, cid in self.name_to_id.items()
        }

        if not self.target_class_ids:
            raise RuntimeError(
                "No se encontraron clases objetivo ('car'/'motorcycle') en el modelo YOLO cargado."
            )

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """Ejecuta inferencia y devuelve detecciones de Supervision filtradas por clases."""
        # Llamada directa al modelo; devuelve una lista de resultados (uno por imagen)
        results = self.model(
            frame,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
            classes=self.target_class_ids,
        )
        result = results[0]

        detections = sv.Detections.from_ultralytics(result)
        if len(detections) == 0:
            return detections

        # Filtrar nuevamente por seguridad
        mask = np.isin(detections.class_id, np.array(self.target_class_ids))
        detections = detections[mask]
        if len(detections) == 0:
            return detections

        # Agregar nombre de clase unificado a data
        class_names = np.array([
            self.id_to_unified_label.get(int(cid), str(cid)) for cid in detections.class_id
        ])
        detections.data["class_name"] = class_names
        return detections
