from qrdet import QRDetector

# model_size can be n, s, m, l
qr_detector = QRDetector(model_size="s")

# half: FP16 quantization (default: False)
# int8: INT8 quantization (default: False)
qr_detector.model.export(format="openvino", half=False, int8=False)
