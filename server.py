import io
import cv2
import uvicorn
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from helper import read_image, detect_common_objects, Model


app = FastAPI(title="Detect Common Objects")

@app.get("/")
def home():
    return "API is working fine, go ahead and detect common objects!"

@app.post("/predict")
def prediction(model: Model, file: UploadFile=File(...)):

    # validate input file
    filename = file.filename
    if not filename.split('.')[-1] in ("jpg", "jpeg", "png"):
        raise HTTPException(status_code=415, detail="Unsupported file type!")

    # get image
    image_stream = io.BytesIO(file.file.read())
    image_stream.seek(0)
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)

    # decode bytes
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # detect
    output_img = detect_common_objects(image, model, save_path=f"./images_detected/{filename}")
    
    # return image
    file_img = open(f"./images_detected/{filename}", "rb")
    return StreamingResponse(file_img, media_type="image/jpeg")


if __name__=="__main__":
    uvicorn.run(app, host="localhost", port=8000)



