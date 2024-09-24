# qrdetector_openvino
QR Code Detection Accelerated by OpenVINO

# Dependencies
* [Ultralytics](https://github.com/ultralytics/ultralytics)
    ```
    pip install ultralytics
    ```
* [OpenVINO](https://github.com/openvinotoolkit/openvino)
    ```
    pip install openvino
    ```
* [ONNX](https://github.com/onnx/onnx)
    ```
    pip install onnx
    ```
* [qrdet](https://github.com/Eric-Canas/qrdet)
    ```
    pip install qrdet
    ```

# Run
```py
from qrdetector_openvino import QRDetectorOpenVINO
import cv2

image = cv2.imread("input.jpg", cv2.IMREAD_COLOR)

qr_detector = QRDetectorOpenVINO(model_size=model_size)
detections = qr_detector.detect(image=image, is_bgr=True)

# Draw the detections
for detection in detections:
    x1, y1, x2, y2 = detection["bbox_xyxy"].astype(int)
    confidence = detection["confidence"]
    cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
    cv2.putText(image, f"{confidence:.2f}", (x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, color=(0, 255, 0), thickness=2)

# Save the results
cv2.imwrite(filename="qr_detections.jpg", img=image)
```
Please see here for complete example [here](https://github.com/ravijo/qrdetector_openvino/blob/main/example.py)

# Export
You can export to any supported format such as ONNX etc. Please see an example [here](https://github.com/ravijo/qrdetector_openvino/blob/main/export.py) that shows exporting to OpenVINO format.

# Reference (and thanks)
* [qrdet](https://github.com/Eric-Canas/qrdet)
