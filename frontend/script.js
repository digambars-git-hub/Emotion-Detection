const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const emotionText = document.getElementById("emotion");
const statusText = document.getElementById("status");

let lastEmotion = "—";
let isProcessing = false;

// CAMERA SETUP
navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => {
    video.srcObject = stream;
    statusText.innerText = "Camera ready. Detecting emotion…";
  })
  .catch(() => {
    statusText.innerText = "Camera access denied.";
  });

// CONFIG (HF-ready)
const API_URL = "/api/predict";  
// later: HF / cloud backend will proxy this

setInterval(async () => {
  if (isProcessing || video.videoWidth === 0) return;
  isProcessing = true;

  const ctx = canvas.getContext("2d");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.drawImage(video, 0, 0);

  canvas.toBlob(async blob => {
    try {
      const formData = new FormData();
      formData.append("file", blob);

      const res = await fetch(API_URL, {
        method: "POST",
        body: formData
      });

      const data = await res.json();
      lastEmotion = data.emotion;
      emotionText.innerText = "Emotion: " + lastEmotion;

    } catch (err) {
      statusText.innerText = "Backend not reachable.";
    }

    isProcessing = false;
  });

}, 1200); // ~1 FPS (stable + cheap)
