import cv2
from deepface import DeepFace
import numpy as np

cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam.")

print("Camera opened. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80,80))

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]

        if face_img.size == 0:
            continue

        face_resized = cv2.resize(face_img, (224, 224))

        try:
            result = DeepFace.analyze(
                img_path = face_resized,
                actions = ['emotion'],
                enforce_detection = False,
                detector_backend = 'opencv'    # ← MOST IMPORTANT
            )
            emotion = result[0]['dominant_emotion']
        except Exception as e:
            print("Emotion error:", e)
            emotion = "err"

        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
