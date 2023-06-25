import time

import torch
from PIL import Image

image_path = 'image.png'
n_iterations = 1000

print("Torch Section")
yoloModel = torch.hub.load('ultralytics/yolov5', 'custom', path="best.pt")

start = time.time()
for _ in range(n_iterations):
    img = Image.open(image_path)
    # display(img)
    results = yoloModel(image_path)
end = time.time()

duration = end - start
print("Duration", duration)
print("Duration per Cycle", duration/n_iterations)
