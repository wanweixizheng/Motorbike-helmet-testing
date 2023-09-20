import os
import cv2
import numpy as np
from detect_motorcyclist import detect_motorcyclist, helmet_detector

video_folder = 'C:/Users/Cesar/Desktop/syde675/input_video'
output_folder = 'C:/Users/Cesar/Desktop/syde675/output_data'
motorcycle_images = []
motorcycle_labels = []

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        frame_count += 1

        if not ret:
            break

        # One frame processed every 3 seconds
        if frame_count % (3*fps) == 0:
            motorcyclist_boxes, helmet_boxes = detect_motorcyclist(frame)

            for motorcyclist_box in motorcyclist_boxes:
                x, y, w, h = motorcyclist_box
                motorcyclist_img = frame[y:y+h, x:x+w]

                # Check that motorcyclists are wearing helmets
                is_wearing_helmet = False
                for helmet_box in helmet_boxes:
                    hx, hy, hw, hh = helmet_box
                    if x <= hx <= x + w and y <= hy <= y + h:
                        is_wearing_helmet = True
                        break

                label = 1 if is_wearing_helmet else 0
                motorcycle_images.append(motorcyclist_img)
                motorcycle_labels.append(label)

    cap.release()

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for video_file in os.listdir(video_folder):
    video_path = os.path.join(video_folder, video_file)
    process_video(video_path)

for i, (image, label) in enumerate(zip(motorcycle_images, motorcycle_labels)):
    output_path = os.path.join(output_folder, f'image_{i}_label_{label}.jpg')
    cv2.imwrite(output_path, image)

np.save(os.path.join(output_folder, 'motorcycle_labels.npy'), motorcycle_labels)