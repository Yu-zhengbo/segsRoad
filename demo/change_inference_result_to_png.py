import numpy as np
from PIL import Image
import os

source_path = './deep/train/vis'
target_path = './deep/train/coarse_mask'

if os.path.exists(target_path) == False:
    os.makedirs(target_path)

for file in os.listdir(source_path):
    if file.endswith('.jpg'):
        img = Image.open(os.path.join(source_path, file))
        img = np.array(img)[:, :, 0]
        img = (img > 0.5).astype(np.uint8)
        img = Image.fromarray(img)
        img.save(os.path.join(target_path, file.replace('.jpg', '.png')))