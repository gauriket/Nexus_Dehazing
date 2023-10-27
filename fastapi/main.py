from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
import shutil
import cv2
import numpy as np
import os
import asyncio

app = FastAPI()

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
    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(dehazed_array)):
        out.write(dehazed_array[i])
    out.release()

def dehaze_video(input_path, output_path):
    frame_array = video2framesarray(input_path)
    dehazed_array = dehaze_images(frame_array)
    dehazed2video(dehazed_array, output_path)

@app.post("/dehaze")
async def dehaze_endpoint(file: UploadFile = File(...), fps: int = Form(...)):
    if file.content_type != "video/mp4":
        return {"error": "Only MP4 videos are supported."}

    input_video = "temp_video.mp4"
    output_video = "output_video1.mp4"  # You can change the format as needed

    with open(input_video, "wb") as f:
        shutil.copyfileobj(file.file, f)

    await asyncio.to_thread(dehaze_video, input_video, output_video)

    return FileResponse(output_video, media_type="video/mp4")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)