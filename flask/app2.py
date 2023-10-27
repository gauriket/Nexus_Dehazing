from flask import Flask, request, send_file, render_template
import cv2
import numpy as np
import os

app = Flask(__name__)

def video2framesarray(videoinput):
    cap = cv2.VideoCapture(videoinput)
    frame_number = 0
    frame_array = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_array.append(frame)
        frame_number += 1
    cap.release()
    return frame_array

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

def dehazed2video(dehazed_array, pathOut, fps=30):
    size = (dehazed_array[0].shape[1], dehazed_array[0].shape[0])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(pathOut, fourcc, fps, size)

    for i in range(len(dehazed_array)):
        out.write(dehazed_array[i])
    out.release()

def record_and_dehaze_video(output_path, duration=10, fps=30):
    cap = cv2.VideoCapture(0)  # 0 represents the default camera
    frames = []
    start_time = cv2.getTickCount()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        current_time = cv2.getTickCount()
        if (current_time - start_time) / cv2.getTickFrequency() >= duration:
            break
    cap.release()

    dehazed_array = dehaze_images(frames)
    dehazed2video(dehazed_array, output_path, fps)

@app.route("/")
def load_page():
    return render_template('app.html')

@app.route("/dehaze", methods=["POST"])
def dehaze_endpoint():
    output_video = "output_video1.mp4"

    record_and_dehaze_video(output_video)

    return send_file(output_video, mimetype="video/mp4")

if __name__ == '__main__':
    app.debug = True
    app.run(host="0.0.0.0", port=5000)
