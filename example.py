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

