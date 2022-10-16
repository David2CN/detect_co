import cv2
import torch
import numpy as np
from enum import Enum
from torch.hub import load


class Model(str, Enum):
    yolov5small = "yolov5s"
    yolov5nano = "yolov5n"


def read_image(img_path: str):
    """
    reads in an image from path and returns a numpy.ndarray
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_model(model_name: str="yolo5s"):
    """
    loads required 
    model_name = "yolov5n" or "yolov5s"
    """
    model_path = f"./models/{model_name}.pt"
    model = torch.hub.load('ultralytics/yolov5', "custom", model_path)
    return model


def draw_bboxes(img: np.ndarray, results: dict, thresh: float=0.5, color: tuple=(0, 0, 255)):
    """
    draws the bounding boxes and labels on the image
    """
    x_shape, y_shape = img.shape[1], img.shape[0]
    for object, result in results.items():
        confidence = result["confidence"]
        if confidence >= thresh:
            x1, y1 = int(result["xmin"]), int(result["ymin"])
            x2, y2 = int(result["xmax"]), int(result["ymax"])

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{result['name']}-{confidence:.2f}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return img


def detect_common_objects(img: np.ndarray, model: str="yolo5s", save_path: str="image.jpg", 
                        thresh: float=0.5, color: tuple=(0, 0, 255), show=False):
    """
    detect common objects in an image and return a new image showing the bounding
    boxes and labels of detected objects.
    """
    # load model
    model = load_model(model)
    model.eval()
    
    # detect
    results = model(img)
    predictions = results.pandas().xyxy[0]
    print(f"Results: {predictions}")

    results = predictions.T.to_dict()
    if results != {}:
        img2 = draw_bboxes(img, results, thresh=thresh, color=color)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        cv2.imwrite(save_path, img2)
        if show:
            cv2.imshow("Object", img2)
            cv2.waitKey(0)
        
        return img2
    
    else:
        return img