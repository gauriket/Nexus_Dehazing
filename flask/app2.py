from flask import Flask, request, send_file, render_template
import cv2
import numpy as np
import os
import shutil
from os.path import isfile
from ultralytics import YOLO

app = Flask(__name__)

def video2framesarray(videoinput):
    cap = cv2.VideoCapture(videoinput)
    frame_number = 0
    frame_array = []
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_array.append(frame)
        frame_number += 1

        if frame_number == 1:
            start_time = cv2.getTickCount()
        
    end_time = cv2.getTickCount()
    time = (end_time - start_time) / cv2.getTickFrequency()
    fps = frame_number / (time * 10)

    cap.release()
    return frame_array, fps

def atmosdehaze(frame):
    # Convert the image to floating point representation
    hazy_image = frame.astype(np.float32) / 255.0

    # Estimate the atmospheric light
    dark_channel = np.min(hazy_image, axis=2)
    atmospheric_light = np.percentile(dark_channel, 99)

    # Estimate the transmission map
    transmission = 1 - 0.95 * dark_channel / (atmospheric_light + 1e-6)  # Add a small epsilon value

    # Clamp the transmission values to [0, 1]
    transmission = np.clip(transmission, 0, 1)

    # Estimate the scene radiance
    scene_radiance = np.zeros_like(hazy_image)
    for channel in range(3):
        scene_radiance[:, :, channel] = (hazy_image[:, :, channel] - atmospheric_light) / (transmission + 1e-6) + atmospheric_light  # Add a small epsilon value

    # Clamp the scene radiance values to [0, 1]
    scene_radiance = np.clip(scene_radiance, 0, 1)

    # Convert the scene radiance back to 8-bit representation
    scene_radiance = (scene_radiance * 255).astype(np.uint8)
    return scene_radiance

def dehaze_images(frame_array):
    dehazed_frames = []
    for frame in frame_array:
        dehazed_frame = atmosdehaze(frame)
        dehazed_frames.append(dehazed_frame)

    # Convert the dehazed frames to an array
    dehazed_array = np.array(dehazed_frames)
    return dehazed_array

def objectdectect(source_images):
    output_folder = 'temp_images'
    os.makedirs(output_folder, exist_ok=True)

    # Save the images from the array to the folder
    for i, image in enumerate(source_images):
        image_filename = os.path.join(output_folder, f'image_{i}.jpg')
        cv2.imwrite(image_filename, image)

    model = YOLO('yolov8n.pt')  # Pretrained YOLOv8n model
    model(source='temp_images', save=True, project='Project')

    # Create an empty list to store the new images with bounding boxes
    output_images = []

    # Define the directory where the saved results are located
    project_directory = 'project'

    # List the files in the 'project/predict' directory
    result_files = os.listdir(os.path.join('Project/predict'))

    # Sort the image files by name
    sorted_result_files = sorted(result_files, key=lambda x: int(os.path.splitext(x)[0].split('_')[1]))

    # Load the sorted image files into an array
    output_images = [cv2.imread(os.path.join('Project/predict', filename)) for filename in sorted_result_files]

    shutil.rmtree('Project')
    return output_images

def dehazed2video(dehazed_array, pathOut, fps=40):
    size = (dehazed_array[0].shape[1], dehazed_array[0].shape[0])
    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(dehazed_array)):
        out.write(dehazed_array[i])
    out.release()

@app.route("/")
def load_page():
    return render_template('app2.html')

@app.route("/dehaze", methods=["POST"])
def dehaze_endpoint():
    file = request.files["file"]
    enable_object_detection = request.form.get("object_detection") == "on"

    if file and file.filename != '' and file.filename.endswith(".mp4"):
        input_video = "temp_video.mp4"
        output_video = "output_video1.mp4"

        file.save(input_video)

        frame_array, fps = video2framesarray(input_video)

        dehazed_array = dehaze_images(frame_array)

        if enable_object_detection:
            source_images = dehazed_array
            output_images = objectdectect(source_images)
            dehazed2video(output_images, output_video, fps)
        else:
            dehazed2video(dehazed_array, output_video, fps)

        return send_file(output_video, mimetype="video/mp4")
    else:
        return {"error": "Only MP4 videos are supported."}

if __name__ == '__main__':
    app.debug = True
    app.run(host="0.0.0.0", port=5000)
