from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io

from inference import predict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/")
def root():
    return {"status": "API running"}

@app.post("/api/predict")
async def predict_emotion(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    emotion = predict(image)
    return {"emotion": emotion}

# Mounted frontend at last
app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")