import io
from PIL import Image

from yolov8 import get_model, detection, segmentation, classification, poseEstimation
from fastapi import FastAPI, File
from starlette.responses import Response


model_det = get_model("detect")
model_seg = get_model("segment")
model_cls = get_model("classify")
model_pose = get_model("pose")

app = FastAPI()

@app.get("/")
def ping():
    return "Hello World"

@app.post("/detect")
def detect(file: bytes = File(...)):
    detected_image = Image.fromarray(detection(model_det, file))
    bytes_io = io.BytesIO()
    detected_image.save(bytes_io, format="PNG")

    return Response(bytes_io.getvalue(), media_type="image/png")

@app.post("/segment")
def segment(file: bytes = File(...)):
    segmented_image = Image.fromarray(segmentation(model_seg, file))
    bytes_io = io.BytesIO()
    segmented_image.save(bytes_io, format="PNG")

    return Response(bytes_io.getvalue(), media_type="image/png")

@app.post("/classify")
def classify(file: bytes = File(...)):
    classified_image = Image.fromarray(classification(model_cls, file))
    bytes_io = io.BytesIO()
    classified_image.save(bytes_io, format="PNG")

    return Response(bytes_io.getvalue(), media_type="image/png")

@app.post("/pose")
def pose(file: bytes = File(...)):
    estimated_image = Image.fromarray(poseEstimation(model_pose, file))
    bytes_io = io.BytesIO()
    estimated_image.save(bytes_io, format="PNG")

    return Response(bytes_io.getvalue(), media_type="image/png")