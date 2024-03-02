# streamlit-fastapi-yolov8

This is a simple web app project serving YOLOv8 models using streamlit and fastapi.

In this project, [YOLOv8](https://docs.ultralytics.com/zh/models/yolov8#supported-tasks-and-modes) models are served using `FastAPI` for the backend service and `streamlit` for the frontend service. There are 4 computer vision tasks that the users can choose: object detection, inastance segmentation, image classification, and pose estimation.


After cloning the repository, create a virtual environment then active it:
```
python -m venv env_name
source env_name/bin/activate
```

Install the required python libraries:
```
pip install -r requirements.txt
```

Visit the following links to download the pretrained models: [YOLOv8n.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt), [YOLOv8n-cls.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-cls.pt), [YOLOv8n-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-seg.pt), [YOLOv8n-pose.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-pose.pt). Create a folder (./models) and move the downloaded models into the folder.

For running the streamlit server, we need to run the following command:
```
cd ./frontend
streamlit run ui.py 
```

For running the fastapi server, we need to open a new terminal, activate the virtual environment, then run the following command:
```
cd ./backend
uvicorn server:app --reload
```

To visit the streamlit UI, visit http://localhost:8501. To visit the FastAPI documentation of the resulting service, visit http://127.0.0.1:8000/docs.

References:
https://davidefiocco.github.io/streamlit-fastapi-ml-serving/