# advanced_demo.py
import time
import cv2
import numpy as np
import os
import requests
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
from deepface import DeepFace
from dotenv import load_dotenv

# -------------------------------------
# Load environment variables
# -------------------------------------
load_dotenv()

DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
ALERT_SAVE_DIR = os.getenv("ALERT_SAVE_DIR", "alerts").strip()
LOG_CSV = os.getenv("LOG_CSV", "events_log.csv").strip()

os.makedirs(ALERT_SAVE_DIR, exist_ok=True)

# -------------------------------------
# Load Haar Cascade for FACE detection
# -------------------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -------------------------------------
# Centroid Tracker
# -------------------------------------
class CentroidTracker:
    def __init__(self, max_lost=20, max_distance=120):
        self.next_id = 0
        self.objects = {}      
        self.bboxes = {}       
        self.lost = {}         
        self.history = {}      
        self.max_lost = max_lost
        self.max_distance = max_distance

    def register(self, centroid, bbox):
        oid = self.next_id
        self.next_id += 1
        self.objects[oid] = centroid
        self.bboxes[oid] = bbox
        self.lost[oid] = 0
        self.history[oid] = [(time.time(), centroid, bbox)]
        return oid

    def deregister(self, oid):
        if oid in self.objects:
            del self.objects[oid]
            del self.bboxes[oid]
            del self.lost[oid]

    def update(self, rects):
        if len(rects) == 0:
            for oid in list(self.lost.keys()):
                self.lost[oid] += 1
                if self.lost[oid] > self.max_lost:
                    self.deregister(oid)
            return self.bboxes

        centroids = []
        for (x, y, w, h) in rects:
            centroids.append((int(x + w/2), int(y + h/2)))

        if len(self.objects) == 0:
            for i in range(len(rects)):
                self.register(centroids[i], rects[i])

        else:
            object_ids = list(self.objects.keys())
            obj_centroids = list(self.objects.values())

            D = np.zeros((len(obj_centroids), len(centroids)))
            for i in range(len(obj_centroids)):
                for j in range(len(centroids)):
                    D[i, j] = np.linalg.norm(np.array(obj_centroids[i]) - np.array(centroids[j]))

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)

            used_rows = set()
            used_cols = set()

            for row in rows:
                col = cols[row]
                if row in used_rows or col in used_cols:
                    continue
                if D[row][col] > self.max_distance:
                    continue

                oid = object_ids[row]
                self.objects[oid] = centroids[col]
                self.bboxes[oid] = rects[col]
                self.history[oid].append((time.time(), centroids[col], rects[col]))
                self.lost[oid] = 0

                used_rows.add(row)
                used_cols.add(col)

            for j in range(len(centroids)):
                if j not in used_cols:
                    self.register(centroids[j], rects[j])

            for i in range(len(obj_centroids)):
                if i not in used_rows:
                    oid = object_ids[i]
                    self.lost[oid] += 1
                    if self.lost[oid] > self.max_lost:
                        self.deregister(oid)

        return self.bboxes


# -------------------------------------
# Behavior Detection
# -------------------------------------
def compute_speed(history, sec=1.0):
    if len(history) < 2:
        return 0
    now = time.time()
    recent = [h for h in history if now - h[0] <= sec]
    if len(recent) < 2:
        return 0
    _, c1, _ = recent[0]
    _, c2, _ = recent[-1]
    dist = np.linalg.norm(np.array(c2) - np.array(c1))
    dt = recent[-1][0] - recent[0][0]
    return dist / max(dt, 0.001)


def detect_fall(history):
    if len(history) < 3:
        return False
    _, (cx1, cy1), (_, _, _, h1) = history[-2]
    _, (cx2, cy2), (_, _, _, h2) = history[-1]
    return (cy2 - cy1) > 60 and h2 < 0.75 * h1


# -------------------------------------
# Alerts
# -------------------------------------
def send_discord_alert(url, message, image_path=None):
    if not url:
        print("[No webhook configured]")
        return
    data = {"content": message}
    if image_path and os.path.exists(image_path):
        with open(image_path, "rb") as f:
            files = {"file": (os.path.basename(image_path), f.read())}
        requests.post(url, data=data, files=files)
    else:
        requests.post(url, json=data)
    print("Alert sent.")


# -------------------------------------
# CSV Logging
# -------------------------------------
def log_event(oid, bbox, emotion, event):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame([{
        "timestamp": ts,
        "id": oid,
        "x": bbox[0],
        "y": bbox[1],
        "w": bbox[2],
        "h": bbox[3],
        "emotion": emotion,
        "event": event
    }])
    header = not os.path.exists(LOG_CSV)
    df.to_csv(LOG_CSV, mode="a", header=header, index=False)


# -------------------------------------
# Load YOLO (person only)
# -------------------------------------
yolo = YOLO("yolov8n.pt")
PERSON_CLASS = 0

# -------------------------------------
# Main Program
# -------------------------------------
cap = cv2.VideoCapture(0)
tracker = CentroidTracker()

frame_count = 0
fps_t0 = time.time()
fps = 0.0   # FIXED

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # YOLO detect persons
    results = yolo.predict(frame, conf=0.4, iou=0.45, imgsz=640, verbose=False)
    rects = []

    for r in results:
        for b in r.boxes:
            cls = int(b.cls[0])
            if cls != PERSON_CLASS:
                continue
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            rects.append((x1, y1, x2 - x1, y2 - y1))

    tracked = tracker.update(rects)

    for oid, bbox in tracked.items():
        x, y, w_box, h_box = bbox
        history = tracker.history[oid]

        speed = compute_speed(history)
        fall = detect_fall(history)

        emotion = "unknown"

        # ---------- FIXED EMOTION DETECTION ----------
        person_crop = frame[y:y+h_box, x:x+w_box]
        gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        if len(faces) > 0:
            (fx, fy, fw, fh) = faces[0]
            face_img = person_crop[fy:fy+fh, fx:fx+fw]

            try:
                face_resized = cv2.resize(face_img, (224, 224))
                out = DeepFace.analyze(face_resized,
                                       actions=["emotion"],
                                       enforce_detection=False,
                                       detector_backend="opencv")
                emotion = out[0]["dominant_emotion"]
            except:
                emotion = "err"
        else:
            emotion = "no-face"

        # Draw person box
        cv2.rectangle(frame, (x, y), (x+w_box, y+h_box), (0,255,0), 2)
        cv2.putText(frame, f"ID:{oid} {emotion} {int(speed)}px/s",
                    (x, max(20, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255,255,255), 2)

        # Running alert
        if speed > 200:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"{ALERT_SAVE_DIR}/run_{oid}_{ts}.jpg"
            cv2.imwrite(path, frame)
            send_discord_alert(DISCORD_WEBHOOK, f"Running detected (ID {oid}) Emotion={emotion}", path)
            log_event(oid, bbox, emotion, "running")

        # Fall alert
        if fall:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"{ALERT_SAVE_DIR}/fall_{oid}_{ts}.jpg"
            cv2.imwrite(path, frame)
            send_discord_alert(DISCORD_WEBHOOK, f"Fall detected (ID {oid}) Emotion={emotion}", path)
            log_event(oid, bbox, emotion, "fall")

    # FPS update
    dt = time.time() - fps_t0
    if dt >= 1:
        fps = frame_count / dt
        fps_t0 = time.time()
        frame_count = 0

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Advanced Surveillance System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
