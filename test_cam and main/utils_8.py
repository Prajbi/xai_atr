import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO, RTDETR

class BaseAIClassifier:
    """Base class containing shared utilities for image processing and annotation."""
    def __init__(self):
        self.model = None

    def _prepare_image(self, pil_image):
        img_array = np.array(pil_image)
        return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    def draw_detections(self, pil_image, detections, box_color):
        cv_image = self._prepare_image(pil_image)
        CONF_THRESHOLD = 0.5
        COLOR_LOW_CONF = (0, 0, 255) # Red

        for det in detections:
            x1, y1, x2, y2 = map(int, det['box'])
            conf = det['conf']
            label = det['label']
            
            current_color = COLOR_LOW_CONF if conf < CONF_THRESHOLD else box_color

            # Draw Box
            cv2.rectangle(cv_image, (x1, y1), (x2, y2), current_color, 2)
            
            # Draw Label Background
            text = f"{label} {conf:.2f}"
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(cv_image, (x1, y1 - th - baseline - 5), (x1 + tw, y1), current_color, -1)
            
            # Draw Text
            cv2.putText(cv_image, text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

class AircraftClassifier(BaseAIClassifier):
    def __init__(self):
        super().__init__()
        self.model_path = 'models/yolo_aircraft.pt'
        self.model = YOLO(self.model_path)

    def _get_detections(self, cv_image):
        # Aircraft logic: Return ALL detections from YOLO
        results = self.model(cv_image, verbose=False)[0]
        detections = []
        for box in getattr(results, "boxes", []):
            detections.append({
                'conf': box.conf.item(),
                'cls': int(box.cls.item()),
                'box': box.xyxy[0].tolist(),
                'label': self.model.names.get(int(box.cls.item()), "Unknown")
            })
        return detections

    def predict(self, pil_image):
        cv_image = self._prepare_image(pil_image)
        dets = self._get_detections(cv_image)
        results = [(d['label'], d['conf']) for d in dets]
        results.sort(key=lambda x: x[1], reverse=True)
        return results if results else [("No Aircrafts Detected", 0.0)]

    def get_annotated_image(self, pil_image):
        cv_image = self._prepare_image(pil_image)
        dets = self._get_detections(cv_image)
        return self.draw_detections(pil_image, dets, (60, 60, 60)) # Dark Grey


class ShipClassifier(BaseAIClassifier):
    def __init__(self):
        super().__init__()
        self.yolo_path = 'models/yolo.pt'
        self.rtdetr_path = 'models/rtdetr.pt'
        self.yolo_model = YOLO(self.yolo_path)
        self.rtdetr_model = RTDETR(self.rtdetr_path)

    def _get_detections(self, cv_image):
        # Ship logic: Ensemble YOLO + RT-DETR and take the BEST per class
        all_candidates = []
        
        # YOLO results
        y_res = self.yolo_model(cv_image, verbose=False)[0]
        for box in getattr(y_res, "boxes", []):
            all_candidates.append({
                'conf': box.conf.item(),
                'cls': int(box.cls.item()),
                'box': box.xyxy[0].tolist(),
                'label': self.yolo_model.names.get(int(box.cls.item()), "Unknown")
            })
        
        # RT-DETR results
        r_res = self.rtdetr_model(cv_image, verbose=False)[0]
        for box in getattr(r_res, "boxes", []):
            all_candidates.append({
                'conf': box.conf.item(),
                'cls': int(box.cls.item()),
                'box': box.xyxy[0].tolist(),
                'label': self.yolo_model.names.get(int(box.cls.item()), "Unknown")
            })

        # Filter for best detection per class
        best_detections = {}
        for det in all_candidates:
            cls_id = det['cls']
            if cls_id not in best_detections or det['conf'] > best_detections[cls_id]['conf']:
                best_detections[cls_id] = det
        return list(best_detections.values())

    def predict(self, pil_image):
        cv_image = self._prepare_image(pil_image)
        dets = self._get_detections(cv_image)
        results = [(d['label'], d['conf']) for d in dets]
        results.sort(key=lambda x: x[1], reverse=True)
        return results if results else [("No Maritime Object Detected", 0.0)]

    def get_annotated_image(self, pil_image):
        cv_image = self._prepare_image(pil_image)
        dets = self._get_detections(cv_image)
        return self.draw_detections(pil_image, dets, (150, 50, 0)) # Dark Blue
