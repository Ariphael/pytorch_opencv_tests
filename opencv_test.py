import numpy as np
import cv2 as cv
import torch
import queue
import time

from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor, as_completed

# Batch size and thread pool configuration
BATCH_SIZE = 2
MAX_WORKERS = 4

cap = cv.VideoCapture(0)
fourcc = cv.VideoWriter_fourcc(*'XVID')
# out = cv.VideoWriter('test.avi', fourcc, 20.0, (640, 480))
device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
model = YOLO("yolo11n.pt").to(device)

frame_queue = queue.Queue()

processing_count = 0
processing_start_time = time.time()

def process_batch(frames):
    frame_tensors = []
    for frame in frames:
        frame_tensor = (
            torch.from_numpy(cv.resize(frame, (640, 480)))
            .permute(2, 0, 1)
            .float()
            .to(device)
        )
        frame_tensors.append(frame_tensor / 255.0)
    
    batch_tensor = torch.stack(frame_tensors)
    results = model(batch_tensor)
    return results

if not cap: 
    print("failed to open camera")
    exit()

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break
        
        frame_queue.put(frame)
        
        frame_resized = cv.resize(frame, (640, 480))
        
        if frame_queue.qsize() >= BATCH_SIZE:
            frames_to_process = []

            for _ in range(BATCH_SIZE):
                try:
                    frames_to_process.append(frame_queue.get_nowait())
                except queue.Empty:
                    break

            futures.append(executor.submit(process_batch, frames_to_process))
            processing_count += len(frames_to_process)

        for future in as_completed(futures):
            results = future.result()
            for result in results:
                frame = result.plot()
                # out.write(frame)
                cv.imshow('frame', frame)

        if cv.waitKey(1) == ord('q'):
            break

        if time.time() - processing_start_time >= 1.0:
            processing_rate = processing_count / (time.time() - processing_start_time) 
            print(f"Processing rate: {processing_rate:.3f} FPS")
            processing_start_time = time.time()
            processing_count = 0
cap.release()
out.release()
cv.destroyAllWindows()

