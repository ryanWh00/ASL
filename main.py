from ultralytics import YOLO

model = YOLO("yolov8n.yaml")
results = model.train(data="/Users/ryanoliver/PycharmProjects/AS_L/American Sign Language Letters.v5-test-quantized-weights.yolov8/data.yaml", epochs=100)
