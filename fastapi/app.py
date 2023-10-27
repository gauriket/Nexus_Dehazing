from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
from tensorflow import keras

app = FastAPI()

# Load your model
model = keras.models.load_model('dehazing_model.h5')
# Define the preprocessing function for dehazing
def preprocess_image(image, model_input_shape):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Ensure RGB format
    image = cv2.resize(image, (model_input_shape[2], model_input_shape[1]))  # Resize to match the model's input shape
    image = image / 255.0  # Normalize pixel values to the range [0, 1]
    return image

# Define the post-processing function for dehazing
def postprocess_image(image):
    image = (image * 255).astype(np.uint8)
    return image

@app.post("/dehaze/")
async def dehaze_image(file: UploadFile):
    # Read the uploaded image
    image_data = await file.read()
    nparr = np.fromstring(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Preprocess the image
    processed_image = preprocess_image(image, model.input_shape)

    # Predict the dehazed image
    dehazed_frame = model.predict(np.array([processed_image]))[0]

    # Post-process the dehazed image
    dehazed_frame = postprocess_image(dehazed_frame)

    # Convert to bytes to return as response
    _, img_encoded = cv2.imencode('.jpg', dehazed_frame)
    return img_encoded.tobytes()
