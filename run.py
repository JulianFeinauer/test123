import os
import time

import wandb
import torch
from PIL import Image

API_KEY = os.environ["API_KEY"]

wandb.login(key=API_KEY)
run = wandb.init()
artifact = run.use_artifact('n-mallouli/Yolov5/run_iy2du03r_model:v0', type='model')
artifact_dir = artifact.download()
weights_path = f"{artifact_dir}/best.pt"
yoloModel = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)

print("Starting Inference...")

start = time.time()
image_path = 'image.png'
img = Image.open(image_path)
# display(img)
results = yoloModel(image_path)
end = time.time()

print('results', results)
print("Duration", (end - start))
# results.show()
