import torch
from pathlib import Path
from helper import read_image, detect_common_objects


img_dir = "./images/"
save_dir = "./images_detected/"

p = Path(img_dir)

model = "yolov5n"  
# model = "yolov5s"

for img_path in p.glob("*.jpg"):
    print(img_path)
    img = read_image(str(img_path))
    detect_common_objects(img, model, save_path=save_dir+img_path.name, show=True)

