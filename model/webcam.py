# webcam.py
import cv2
from PIL import Image
from inference import predict
from collections import deque

emotion_buffer = deque(maxlen=7)

# load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        pad = int(0.2 * w)

        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)

        face = frame[y1:y2, x1:x2]

        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face_rgb = cv2.cvtColor(face_gray, cv2.COLOR_GRAY2RGB)

        pil_face = Image.fromarray(face_rgb)

        emotion_buffer.append(predict(pil_face))
        emotion = max(set(emotion_buffer), key=emotion_buffer.count)

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(
            frame,
            emotion,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0,255,0),
            2
        )

    cv2.imshow("Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
