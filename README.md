# 😊 Emotion Recognition – Real‑Time Facial Emotion Detection

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Latest-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

<div align="center">
  <img src="https://via.placeholder.com/800x400/4A90E2/FFFFFF?text=Emotion+Recognition+Demo" alt="Emotion Recognition Demo" width="800"/>
  
  *Real-time facial emotion detection using deep learning*
</div>

---

This project is a **real‑time facial emotion recognition system** built around a **fine‑tuned MobileNetV2 model**.  
We start from an **ImageNet‑pretrained MobileNetV2**, fine‑tune it on a custom facial emotion dataset, save the weights as `emotion_recognition_model.pth`, and then use that model behind a **FastAPI backend** and **browser-based webcam UI** to detect emotions such as 😠 *Angry*, 😨 *Fear*, 😊 *Happy*, 😢 *Sad,* and 😲 *Surprise* from a live camera feed.

---

## ✨ Features

- 🤖 **Fine‑tuned deep learning model** – MobileNetV2 is fine‑tuned on a labeled facial emotion dataset to specialize it for emotion recognition.
- 📹 **Real‑time detection** from your webcam (browser or local OpenCV).
- 🚀 **FastAPI REST API** endpoint for image-based emotion prediction.
- 🎨 **Simple HTML/JS frontend** that streams frames to the backend and displays the predicted emotion.
- 💻 **Standalone webcam script** (`webcam.py`) if you prefer running everything locally without the browser.

---

## 📁 Project Structure

```
📦 Emotion Recognition Project
├── 📂 frontend/
│   ├── 📄 index.html          # Minimal UI with a video element and live emotion text
│   ├── 📄 script.js            # Grabs webcam frames, sends them to the backend
│   └── 📄 style.css            # Basic styling for the page and video element
├── 📂 model/
│   ├── 🐍 api.py               # FastAPI app exposing POST /api/predict
│   ├── 🐍 inference.py         # Loads model and defines predict(image) function
│   ├── 🐍 model.py             # MobileNetV2 architecture definition
│   ├── 🐍 webcam.py            # OpenCV-based real‑time emotion recognition
│   ├── 📓 working.ipynb        # Jupyter notebook for fine‑tuning MobileNetV2
│   ├── 💾 emotion_recognition_model.pth  # Fine‑tuned PyTorch checkpoint
│   └── 📂 Data/                # Dataset folders (Angry, Fear, Happy, Sad, Surprise)
├── 🐳 DockerFile               # Docker image definition for deployment
└── 📋 requirements.txt         # Python dependencies
```

---

## 🛠️ Requirements

### System Requirements
- 🐍 **Python** 3.8+ (recommended)
- 📦 **pip** for installing dependencies
- 📷 A **webcam**
- ⚡ (Optional) **GPU with CUDA** for faster inference, otherwise CPU will be used.

### Python Packages
Python packages (also listed in `requirements.txt`):

- `torch`, `torchvision` 🔥
- `numpy` 🔢
- `opencv-python` 📹
- `matplotlib` 📊
- `pillow` 🖼️
- `tqdm` 📈
- `requests` 🌐
- `fastapi`, `uvicorn` ⚡ (install explicitly if missing)

### Installation

You can install everything with:

```bash
pip install -r requirements.txt fastapi uvicorn
```

---

## 🚀 How to Run

### 🌐 Web API + Frontend

1. **Navigate to the model folder**:

   ```bash
   cd model
   ```

2. **Start the FastAPI server** (with Uvicorn):

   ```bash
   uvicorn api:app --host 0.0.0.0 --port 8000 --reload
   ```

3. **Open the frontend**:
   - Option A: The API mounts the `frontend` folder as static files.  
     Open your browser and go to:

     ```text
     http://localhost:8000/
     ```

   - If needed, ensure the working directory is such that `../frontend` (from `model/api.py`) points to the `frontend` folder in this repo.

4. **Allow camera access** 📷 in the browser when prompted.

5. You should see the **live video** and the **predicted emotion** updating underneath! 🎉

---

### 💻 Local Webcam Script (No Browser)

If you prefer a pure Python / OpenCV pipeline:

1. Make sure dependencies are installed:

   ```bash
   pip install -r requirements.txt
   ```

2. From the `model` directory, run:

   ```bash
   python webcam.py
   ```

3. A window called **"Emotion Recognition"** will appear:
   - 👤 Detected faces will be highlighted with a bounding box.
   - 🏷️ The predicted emotion label will be shown next to each face.
   - Press **`q`** to quit.

---

### 🐳 Run with Docker (Optional)

If you prefer not to manage Python and system dependencies manually, you can use the provided `DockerFile`:

1. **Build the image** from the project root:

   ```bash
   docker build -t emotion-recognition -f DockerFile .
   ```

2. **Run the container** and expose the API on port `8000`:

   ```bash
   docker run --rm -p 8000:8000 emotion-recognition
   ```

3. Open your browser at:

   ```text
   http://localhost:8000/
   ```

   and allow camera access. The behavior is the same as in the "Web API + Frontend" section above.

---

## 🧠 Model & Inference Details

- The model is a **MobileNetV2** classifier whose final layer is adapted to the number of emotion classes (e.g. 😠 Angry, 😨 Fear, 😊 Happy, 😢 Sad, 😲 Surprise).
- The **fine‑tuned weights and class names** are stored in `emotion_recognition_model.pth`, which is produced by the training notebook (`working.ipynb`).
- Images are preprocessed with:
  - 📐 Resize to \(224 \times 224\)
  - 🔄 Conversion to tensor
  - 📊 Normalization with ImageNet mean and std
- Inference is done by `inference.py` via:
  - `predict(pil_image)` → returns a string label, e.g. `"Happy"`.

### Testing the Model

You can test the model directly with a static image (from the `model` directory):

```bash
python inference.py
```

This will load `Image.jpg` and print the predicted emotion.

---

## 📡 API Reference

### Endpoints

- **`GET /`**
  - Returns a simple JSON status: `{"status": "API running"}`.

- **`POST /api/predict`**
  - **Body**: `multipart/form-data` with a single field:
    - `file`: image file (e.g. JPEG/PNG).
  - **Response**:
    ```json
    { "emotion": "Happy" }
    ```

The frontend uses this endpoint to send frames from your webcam (as blobs) approximately once per second.

---

## 📚 Dataset & Training (Fine‑Tuning Overview)

- The (optional) `Data` directory is expected to contain labeled images organized by emotion:
  - 😠 `Angry/`, 😨 `Fear/`, 😊 `Happy/`, 😢 `Sad/`, 😲 `Suprise/`.
- The **fine‑tuning process is captured in `working.ipynb`**:
  - Load an **ImageNet‑pretrained MobileNetV2** from `torchvision`.
  - Replace the final classification layer so that its output dimension matches the number of emotion classes.
  - Create PyTorch `Dataset`/`DataLoader` objects from the `Data` folders with standard augmentations and preprocessing.
  - Train the network (typically using cross‑entropy loss and an optimizer like Adam/SGD) while monitoring validation accuracy.
  - Save the best model checkpoint and class mapping to `emotion_recognition_model.pth`.
- You can adapt the model for new datasets by:
  - Updating the `Data` folders and class list.
  - Re‑running or modifying `working.ipynb` and, if needed, adjusting `model.py` for a different backbone or number of classes.

**Dataset Source**: [Kaggle - Human Face Emotions](https://www.kaggle.com/datasets/samithsachidanandan/human-face-emotions)

---

## 🔧 Troubleshooting

- **📷 Camera access denied (browser)**  
  - Check browser permissions and ensure you're using `http://localhost` (not `file://`).

- **🔌 "Backend not reachable" in the frontend**  
  - Confirm the FastAPI server is running on the same host/port that `script.js` expects (`/api/predict` → default `http://localhost:8000/api/predict`).
  - Check for CORS issues or port conflicts.

- **📦 Model file not found**  
  - Ensure `emotion_recognition_model.pth` is present in the `model` directory when running any Python scripts there.

---

## 📄 License & Credits

- The project uses **PyTorch**, **FastAPI**, **OpenCV**, and **PIL** under their respective licenses.
- Dataset images in `Data/` should respect their original source licenses (not provided here).
- Feel free to modify or extend this project for research, learning, or personal use.

---

<div align="center">
  <p>Made with ❤️ using PyTorch, FastAPI, and OpenCV</p>
  <p>⭐ Star this repo if you find it helpful!</p>
</div>
