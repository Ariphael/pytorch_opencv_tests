import time
import cv2 as cv

cap = cv.VideoCapture(0)
fourcc = cv.VideoWriter_fourcc(*'XVID')

frame_count = 0
start_time = time.time()

if not cap:
    print("Failed to open cap")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if time.time() - start_time >= 1.0:
        capture_rate = frame_count / (time.time() - start_time)
        print(f"Capture Rate: {capture_rate:.2f} FPS")
        frame_count = 0
        start_time = time.time()
