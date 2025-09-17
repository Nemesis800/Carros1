from __future__ import annotations
import numpy as np
import supervision as sv
from ultralytics import YOLO
from typing import Dict, List


class VehicleDetector:
    """
    Envuelve un modelo YOLO de Ultralytics para detectar vehículos (car y motorcycle).
    Soporta nombres: yolo12n.pt, yolo11n.pt, yolov8n.pt (en ese orden de preferencia si aplica).
    """

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        conf: float = 0.3,
        iou: float = 0.5,
        device: str | int | None = None,
        classes_whitelist: List[str] | None = None,
    ) -> None:

        # ---- Candidatos de carga (fallbacks) ----
        # Si piden yolo12n.pt, probamos 12 -> 11 -> v8
        # Si piden yolo11n.pt, probamos 11 -> 12 -> v8
        # Si piden yolov8n.pt, probamos v8 -> 11 -> 12
        preferred = [model_name]
        for alt in ["yolo12n.pt", "yolo11n.pt", "yolov8n.pt"]:
            if alt not in preferred:
                preferred.append(alt)

        self.model = None
        last_exc: Exception | None = None
        for cand in preferred:
            try:
                self.model = YOLO(cand)
                # print(f"[INFO] Modelo cargado: {cand}")
                break
            except Exception as e:
                # print(f"[WARN] No se pudo cargar '{cand}': {e}")
                last_exc = e

        if self.model is None:
            raise RuntimeError(
                f"No se pudo cargar ningún modelo de la lista {preferred}. "
                f"Último error: {last_exc}"
            )

        self.conf = conf
        self.iou = iou
        self.device = device

        # ---- Mapeo de clases del modelo ----
        try:
            model_names = self.model.names  # type: ignore[attr-defined]
        except Exception:
            model_names = getattr(self.model.model, "names", {})  # fallback

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
                if cname_l == "motorbike":
                    cname_l = "motorcycle"
                if cname_l not in self.name_to_id:
                    self.name_to_id[cname_l] = cid

        self.target_class_ids: List[int] = list(self.name_to_id.values())
        self.id_to_unified_label: Dict[int, str] = {
            cid: name for name, cid in self.name_to_id.items()
        }

        if not self.target_class_ids:
            raise RuntimeError(
                "No se encontraron clases objetivo ('car'/'motorcycle') en el modelo YOLO cargado."
            )

    def detect(self, frame: np.ndarray) -> sv.Detections:
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

        mask = np.isin(detections.class_id, np.array(self.target_class_ids))
        detections = detections[mask]
        if len(detections) == 0:
            return detections

        class_names = np.array([
            self.id_to_unified_label.get(int(cid), str(cid)) for cid in detections.class_id
        ])
        detections.data["class_name"] = class_names
        return detections

