# advanced_demo_perf.py
# Improved performance version of advanced_demo.py
# - Threaded video capture
# - YOLO cadence (skip frames)
# - Async DeepFace calls with caching per ID
# - Face detection inside person bbox (Haar or cv2.dnn optional)
# Tuning knobs are at the top.

import time
import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
from deepface import DeepFace
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue

load_dotenv()
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
ALERT_SAVE_DIR = os.getenv("ALERT_SAVE_DIR", "alerts").strip()
LOG_CSV = os.getenv("LOG_CSV", "events_log.csv").strip()
os.makedirs(ALERT_SAVE_DIR, exist_ok=True)

# -------------------------
# TUNING PARAMETERS
# -------------------------
IMG_WIDTH = 640             # frame width to resize to (processing size) - lower = faster
YOLO_FRAME_SKIP = 2         # run YOLO every N frames (1 = every frame)
DEEPFACE_WORKERS = 2        # number of threads for DeepFace inference
EMOTION_INTERVAL = 2.0      # seconds between emotion inferences for same ID
DEEPFACE_TIMEOUT = 6.0      # seconds to wait for deepface future before considering it stale
MIN_FACE_SIZE = 50          # minimum face width/height (px) to analyze
YOLO_IMGSZ = 480            # yolov8 imgsz (use small for speed)
YOLO_CONF = 0.4
YOLO_IOU = 0.45
SPEED_RUNNING_THRESHOLD = 180  # pixels/sec for running (tune for your camera)
# -------------------------

# Face detector (Haar cascade; very fast)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Threaded camera class
class CameraStream:
    def __init__(self, src=0, width=IMG_WIDTH):
        self.cap = cv2.VideoCapture(src)
        self.running = True
        self.Q = queue.Queue(maxsize=2)
        _, frame = self.cap.read()
        if frame is None:
            raise RuntimeError("Cannot open camera")
        self.width = width
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                break
            # resize for consistent processing
            h, w = frame.shape[:2]
            if w != self.width:
                new_h = int(h * (self.width / w))
                frame = cv2.resize(frame, (self.width, new_h))
            # put latest frame, dropping old if necessary
            try:
                if self.Q.full():
                    _ = self.Q.get_nowait()
                self.Q.put_nowait(frame)
            except queue.Full:
                pass
        self.cap.release()

    def read(self, timeout=1.0):
        try:
            return self.Q.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        self.thread.join(timeout=1.0)

# Simple centroid tracker (same as before, minimal)
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
            for i in range(len(centroids)):
                self.register(centroids[i], rects[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            D = np.zeros((len(object_centroids), len(centroids)))
            for i in range(len(object_centroids)):
                for j in range(len(centroids)):
                    D[i, j] = np.linalg.norm(np.array(object_centroids[i]) - np.array(centroids[j]))
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)
            used_rows = set()
            used_cols = set()
            for row in rows:
                col = cols[row]
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
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
            for i in range(len(object_centroids)):
                if i not in used_rows:
                    oid = object_ids[i]
                    self.lost[oid] += 1
                    if self.lost[oid] > self.max_lost:
                        self.deregister(oid)
        return self.bboxes

# Behavior heuristics
def compute_speed(history, window_seconds=1.0):
    if len(history) < 2:
        return 0.0
    now = time.time()
    pts = [h for h in history if now - h[0] <= window_seconds]
    if len(pts) < 2:
        pts = history[-2:]
    if len(pts) < 2:
        return 0.0
    (t1, c1, _) = pts[0]
    (t2, c2, _) = pts[-1]
    dist = np.linalg.norm(np.array(c2) - np.array(c1))
    dt = max(1e-3, t2 - t1)
    return dist / dt

def detect_fall(history):
    if len(history) < 3:
        return False
    last = history[-1]
    prev = history[-2]
    _, (cx1, cy1), bbox1 = prev
    _, (cx2, cy2), bbox2 = last
    h1 = bbox1[3]; h2 = bbox2[3]
    dy = cy2 - cy1
    if dy > 60 and h2 < 0.75 * h1:
        return True
    return False

# Alert logging
def log_event(timestamp, oid, bbox, emotion, event_type):
    row = {"timestamp": timestamp, "id": oid, "x": int(bbox[0]), "y": int(bbox[1]),
           "w": int(bbox[2]), "h": int(bbox[3]), "emotion": emotion, "event": event_type}
    df = pd.DataFrame([row])
    header = not os.path.exists(LOG_CSV)
    df.to_csv(LOG_CSV, mode='a', header=header, index=False)

def send_discord_alert(webhook_url, content, image_path=None):
    if not webhook_url:
        return
    data = {"content": content}
    try:
        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as f:
                files = {"file": (os.path.basename(image_path), f.read())}
            requests.post(webhook_url, data=data, files=files)
        else:
            requests.post(webhook_url, json=data)
    except Exception as e:
        print("Discord send failed:", e)

# Load YOLO once
print("Loading YOLO model (this may take a moment)...")
yolo = YOLO("yolov8n.pt")  # ensure yolov8n present
PERSON_CLASS = 0

# Thread pool for DeepFace
executor = ThreadPoolExecutor(max_workers=DEEPFACE_WORKERS)
futures_by_id = {}   # oid -> (future, timestamp_submitted)

# Per-ID cache: {oid: {"emotion":..., "ts":..., "status":...}}
id_cache = {}

# Start camera
cam = CameraStream(src=0, width=IMG_WIDTH)
tracker = CentroidTracker()
frame_count = 0
fps_t0 = time.time()
fps = 0.0

try:
    print("Starting main loop. Press 'q' to quit.")
    while True:
        frame = cam.read(timeout=1.0)
        if frame is None:
            continue
        frame_count += 1
        display = frame.copy()

        # Run YOLO every YOLO_FRAME_SKIP frames to save CPU
        run_yolo = (frame_count % YOLO_FRAME_SKIP == 0)
        rects = []
        if run_yolo:
            results = yolo.predict(frame, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, iou=YOLO_IOU, verbose=False)
            for r in results:
                if not hasattr(r, "boxes"):
                    continue
                for box in r.boxes:
                    cls = int(box.cls[0].item())
                    if cls != PERSON_CLASS:
                        continue
                    xyxy = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = map(int, xyxy)
                    # clamp within frame
                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(frame.shape[1]-1, x2); y2 = min(frame.shape[0]-1, y2)
                    bw = x2 - x1; bh = y2 - y1
                    if bw <= 0 or bh <= 0:
                        continue
                    rects.append((x1, y1, bw, bh))
        else:
            # use previous tracker bboxes as rects to keep moving objects annotated even when skipping YOLO
            rects = list(tracker.bboxes.values())

        tracked = tracker.update(rects)

        # Manage futures: check finished deepface tasks
        completed_ids = []
        for oid, info in list(futures_by_id.items()):
            future, ts_sub = info
            if future.done():
                try:
                    out = future.result(timeout=1.0)
                    # out may be dict or list; normalize
                    emotion = None
                    if isinstance(out, list):
                        emotion = out[0].get("dominant_emotion", "unknown")
                    elif isinstance(out, dict):
                        emotion = out.get("dominant_emotion", "unknown")
                    else:
                        emotion = "unknown"
                except Exception as e:
                    emotion = "err"
                id_cache.setdefault(oid, {})["emotion"] = emotion
                id_cache[oid]["ts"] = time.time()
                id_cache[oid]["status"] = "ready"
                completed_ids.append(oid)
                del futures_by_id[oid]
            else:
                # if a future is too old, cancel and remove
                if time.time() - ts_sub > DEEPFACE_TIMEOUT:
                    try:
                        future.cancel()
                    except:
                        pass
                    del futures_by_id[oid]

        # For each tracked object, possibly submit deepface job
        for oid, bbox in tracked.items():
            x, y, bw, bh = bbox
            history = tracker.history.get(oid, [])
            speed = compute_speed(history)
            fall_flag = detect_fall(history)

            # default emotion from cache
            cache = id_cache.get(oid, {})
            emotion_label = cache.get("emotion", "unknown")
            last_ts = cache.get("ts", 0)

            # Extract person crop for face detection
            # ensure bbox inside image bounds
            person_crop = frame[y:y+bh, x:x+bw]
            if person_crop.size == 0:
                face_found = False
            else:
                gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                face_found = len(faces) > 0

            # Decide whether to run emotion inference:
            need_infer = False
            if face_found:
                # If no cached emotion or enough time passed since last inference, schedule it
                if (time.time() - last_ts) > EMOTION_INTERVAL and oid not in futures_by_id:
                    # check face size
                    (fx, fy, fw, fh) = faces[0]
                    if fw >= MIN_FACE_SIZE and fh >= MIN_FACE_SIZE:
                        need_infer = True

            # Submit DeepFace job asynchronously
            if need_infer:
                # prepare face crop
                fx, fy, fw, fh = faces[0]
                face_img = person_crop[fy:fy+fh, fx:fx+fw]
                try:
                    face_resized = cv2.resize(face_img, (224, 224))
                except:
                    face_resized = face_img
                # submit to executor
                future = executor.submit(DeepFace.analyze, face_resized, ["emotion"], False, "opencv")
                futures_by_id[oid] = (future, time.time())
                id_cache.setdefault(oid, {})["status"] = "running"
                # set interim label so UI shows analyzing
                emotion_label = id_cache[oid].get("emotion", "analyzing")

            # When no face found, keep label as 'no-face'
            if not face_found:
                emotion_label = "no-face"

            # Draw visual info
            label = f"ID:{oid} {emotion_label} {int(speed)}px/s"
            cv2.rectangle(display, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(display, label, (x, max(15, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Running/fall detection and alert (non-blocking)
            if speed > SPEED_RUNNING_THRESHOLD:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                img_name = os.path.join(ALERT_SAVE_DIR, f"run_{oid}_{ts}.jpg")
                cv2.imwrite(img_name, frame)
                log_event(ts, oid, bbox, emotion_label, "running")
                # send_discord_alert in background (optional)
            if fall_flag:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                img_name = os.path.join(ALERT_SAVE_DIR, f"fall_{oid}_{ts}.jpg")
                cv2.imwrite(img_name, frame)
                log_event(ts, oid, bbox, emotion_label, "fall")

        # FPS calc
        dt = time.time() - fps_t0
        if dt >= 1.0:
            fps = frame_count / dt
            frame_count = 0
            fps_t0 = time.time()
        cv2.putText(display, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Advanced Surveillance - Perf Mode", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

finally:
    cam.stop()
    executor.shutdown(wait=False)
    cv2.destroyAllWindows()
