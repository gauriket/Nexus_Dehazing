const express = require("express");
const multer = require("multer");
const { createWriteStream, createReadStream } = require("fs");
const { promisify } = require("util");
const pipeline = promisify(require("stream").pipeline);

const app = express();
const port = 3000;

app.use(express.json());

const tf = require("@tensorflow/tfjs-node");
let model;

async function loadModel() {
  model = await tf.loadLayersModel("file://assets/dehazing_model.h5");
}

loadModel().catch(console.error);

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});

const storage = multer.memoryStorage();
const upload = multer({ storage });

const processVideoWithModel = async (videoBuffer) => {
  const inputVideoPath = "assets/hazed.mp4";

  await pipeline(
    createReadStream("input.mp4"),
    createWriteStream(inputVideoPath)
  );

  const ffmpeg = require("fluent-ffmpeg");
  const outputVideoPath = "assets/output.mp4";

  return new Promise((resolve, reject) => {
    ffmpeg()
      .input(inputVideoPath)
      .inputFormat("mp4")
      .inputFPS(30)
      .output(outputVideoPath)
      .on("end", () => resolve(outputVideoPath))
      .on("error", (err) => reject(err))
      .run();
  });
};

app.post("/dehaze", upload.single("video"), async (req, res) => {
  try {
    if (!model) {
      return res.status(500).send("Model not loaded");
    }

    const videoBuffer = req.file.buffer;

    const dehazedVideoPath = await processVideoWithModel(videoBuffer);

    // Read the processed video and return it as a response
    const dehazedVideoBuffer = await fs.promises.readFile(dehazedVideoPath);

    res.set("Content-Type", "video/mp4");
    res.send(dehazedVideoBuffer);
  } catch (error) {
    console.error(error);
    res.status(500).send("Internal Server Error");
  }
});
