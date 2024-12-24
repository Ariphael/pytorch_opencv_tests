import numpy as np
import cv2 as cv
import torch

from ultralytics import YOLO

cap = cv.VideoCapture(0)
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('test.avi', fourcc, 20.0, (640, 480))
device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
model = YOLO("yolo11n.pt").to(device)

frame_count = 0
frame_skip = 10

if not cap.isOpened():
    print("failed to open camera")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_resized = cv.resize(frame, (640, 480))

    if frame_count % frame_skip == 0:
        frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float().to(device)
        frame_tensor /= 255.0
        results = model(frame_tensor.unsqueeze(0))
        frame = result[0].plot()
    else:
        frame = result[0].plot()

    out.write(frame)   
    cv.imshow('frame', frame)
    
    frame_count += 1

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
cv.destroyAllWindows()

