from qrdet import QRDetector
from ultralytics import YOLO

class QRDetectorOpenVINO(QRDetector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        model_size = kwargs.get("model_size", None)
        self.model = YOLO(f"qrdet-{model_size}_openvino_model/", task="segment")
