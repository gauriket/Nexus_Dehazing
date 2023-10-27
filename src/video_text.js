const cv = require('opencv4nodejs');
const { createReadStream } = require('fs');

const videoPath = 'assets/hazed.mp4'; // Adjust to your video file path

const cap = new cv.VideoCapture(videoPath);

if (!cap.isOpen()) {
  console.error('Error: Could not open video.');
  process.exit(1);
}

const window = new cv.NamedWindow('Video Player', cv.WINDOW_NORMAL);

while (true) {
  const frame = cap.read();
  if (frame.empty) {
    console.log('End of video');
    break;
  }

  window.show(frame);
  const key = window.blockingWaitKey(10);
  if (key === 27) {
    // Exit when the 'Esc' key is pressed
    break;
  }
}

cap.release();
window.destroy();
