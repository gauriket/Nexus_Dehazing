from fastapi import FastAPI, File, UploadFile
from typing import List
import cv2
import numpy as np
from tensorflow import keras

# Initialize the FastAPI app
app = FastAPI()

# Load the trained model for dehazing
model = keras.models.load_model('dehazing_model.h5')

# Define the preprocessing function for dehazing
def preprocess_image(image, model_input_shape):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (model_input_shape[2], model_input_shape[1]))
    image = image / 255.0
    return image

# Define the post-processing function for dehazing
def postprocess_image(image):
    image = (image * 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# Define the dehaze function
def dehaze_image(image):
    dehazed_frame = model.predict(np.array([image]))[0]
    dehazed_frame = postprocess_image(dehazed_frame)
    return dehazed_frame

@app.post("/dehaze")
async def dehaze_video(files: List[UploadFile]):
    frame_array = []

    for uploaded_file in files:
        # Read and preprocess each frame
        contents = await uploaded_file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is not None:
            frame = preprocess_image(frame, model.input_shape)
            dehazed_frame = dehaze_image(frame)
            frame_array.append(dehazed_frame)

    if frame_array:
        # Create the output video
        output_path = 'output_video.avi'
        size = (frame_array[0].shape[1], frame_array[0].shape[0])
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'DIVX'), 30, size)

        for dehazed_frame in frame_array:
            out.write(dehazed_frame)
        out.release()

        return {"message": "Dehazing completed. You can download the dehazed video from the provided link.", "output_video_path": output_path}
    else:
        return {"message": "No frames were processed."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)
