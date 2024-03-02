import io

from PIL import Image
import numpy as np
import cv2

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator



def get_model(task):
    if task == "detect":
        model = YOLO('../models/yolov8n.pt') 
    elif task == "segment":
        model = YOLO('../models/yolov8n-seg.pt') 
    elif task == "classify":
        model = YOLO('../models/yolov8n-cls.pt') 
    elif task == "pose":
        model = YOLO('../models/yolov8n-pose.pt') 
    
    return model

def detection(model, binary_image):
    #input_image = np.array(Image.open(io.BytesIO(binary_image)).convert("RGB"))
    np_arr = np.fromstring(io.BytesIO(binary_image).read(), np.uint8)
    input_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    annotator = Annotator(input_image)

    results = model.predict(input_image, conf=0.5)
    boxes = results[0].boxes
    for box in boxes:
        b = box.xyxy[0] 
        c = box.cls
        annotator.box_label(b, model.names[int(c)], color=(0, 0, 255))
    output_image = annotator.result()  

    return cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

def segmentation(model, binary_image):
    np_arr = np.fromstring(io.BytesIO(binary_image).read(), np.uint8)
    input_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    results = model.predict(input_image, conf=0.5)
    output_image = results[0].plot()

    return cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

def classification(model, binary_image):
    np_arr = np.fromstring(io.BytesIO(binary_image).read(), np.uint8)
    input_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    annotator = Annotator(input_image)
    x, y = 5, 40

    results = model.predict(input_image, conf=0.5)
    names, probs = results[0].names, results[0].probs.data.tolist()
    for i in range(len(probs)):
        if probs[i] > 0.2:
            name = names[i]
            prob = round(probs[i], 4)
            annotator.text([x, y], str(name) + " " + str(prob), 
                        txt_color=(255, 255, 255), anchor='top', box_style=False)
            y += 40
    output_image = annotator.result()  

    return cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)


def poseEstimation(model, binary_image):
    np_arr = np.fromstring(io.BytesIO(binary_image).read(), np.uint8)
    input_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    annotator = Annotator(input_image)

    results = model.predict(input_image, conf=0.5)
    persons = results[0].keypoints
    for person in persons:
        for keypoints in person:
            keypts = keypoints.data[0]
            annotator.kpts(keypts, shape=(640, 640), radius=5, kpt_line=True)
    output_image = annotator.result()  

    return cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)