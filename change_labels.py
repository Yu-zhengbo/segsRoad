import os
from PIL import Image
import numpy as np


path = 'G:/datasets/Massachusetts/tiff/annotations'
to_path = 'G:/datasets/Massachusetts/tiff/annotation'


splits = ['train','val','test']

def change_tif_2_png(image_path):
    img = Image.open(image_path)
    img = np.array(img)/255.0
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    img.save(image_path.replace('.tif','.png').replace('annotations','annotation'))

for split in splits:
    for filename in os.listdir(os.path.join(path,split)):
        if filename.endswith('.tif'):
            change_tif_2_png(os.path.join(path,split,filename))

