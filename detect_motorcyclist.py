import torch
from pathlib import Path
from yolov5 import YOLOv5

# Load the pre-trained YOLOv5s model
model = YOLOv5('C:/Users/Cesar/Desktop/syde675/yolov5/yolov5s.pt', device='cuda:0' if torch.cuda.is_available() else 'cpu')

def detect_motorcyclist(image):
    results = model.predict(image)
    motorcyclist_boxes = []
    helmet_boxes = []

    for det in results.pred[0]:
        x, y, x2, y2, conf, cls = det.tolist()

        if cls == 2:  # Assume category ID 2 is for motorcyclists
            motorcyclist_boxes.append((int(x), int(y), int(x2 - x), int(y2 - y)))
        elif cls == 1:  # Assuming category ID 1 is for helmets
            helmet_boxes.append((int(x), int(y), int(x2 - x), int(y2 - y)))

    return motorcyclist_boxes, helmet_boxes

def helmet_detector(helmet_box):
    # A helmet is considered to be worn if the area of the bounding box of the helmet is greater than 0
    return helmet_box[2] * helmet_box[3] > 0