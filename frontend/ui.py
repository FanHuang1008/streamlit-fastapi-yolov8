import streamlit as st
from requests_toolbelt.multipart.encoder import MultipartEncoder
from PIL import Image
import io
import requests



st.title("Basic YOLOv8 App")

task = st.selectbox("Which task do you want to perform?", 
                    ("Object Detection", "Inastance Segmentation", 
                     "Image Classification", "Pose Estimation"))
st.write("")
st.write("")
image = st.file_uploader("Upload your image here", type=['png', 'jpeg', 'jpg'])

# fastapi endpoint
url = 'http://127.0.0.1:8000'
if task == "Object Detection":
    endpoint = '/detect'
elif task == "Inastance Segmentation":
    endpoint = '/segment'
elif task == "Image Classification":
    endpoint = '/classify'
elif task == "Pose Estimation":
    endpoint = '/pose'

if st.button("Make Prediction"):
    if not image:
        st.write("Please insert an image!")
    else:
        m = MultipartEncoder(fields={'file': ('filename', image, 'image/jpeg')})
        response = requests.post(url + endpoint, data=m, 
                                 headers={'Content-Type': m.content_type},
                                 timeout=8000)
        predicted_image = Image.open(io.BytesIO(response.content)).convert("RGB")
        st.image(predicted_image, caption="Prediction Result", use_column_width=True)


