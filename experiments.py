import cv2
import numpy as np
import time
import os
import queue
import threading
import multiprocessing
import math
from collections import deque, defaultdict
from datetime import datetime
from ultralytics import YOLO
from deepface import DeepFace

# --- CONFIGURATION ---
YOLO_MODEL = "yolov8n-pose.pt"  # Pose model is better for fall detection
LOG_FILE = "surveillance_log.csv"
FALL_ANGLE_THRESHOLD = 45       # Angle from vertical to detect a fall
FALL_VELOCITY_THRESHOLD = 150   # Pixels/sec downward movement
EMOTION_CHECK_INTERVAL = 2.0    # Seconds between emotion updates per person

# --- ASYNC VIDEO CAPTURE ---
class VideoCaptureAsync:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.q = queue.Queue(maxsize=2)
        self.stopped = False
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stopped = True
                continue
            if not self.q.empty():
                try: self.q.get_nowait()
                except queue.Empty: pass
            self.q.put(frame)

    def read(self):
        return self.q.get() if not self.q.empty() else None

    def stop(self):
        self.stopped = True
        self.cap.release()

# --- BACKGROUND EMOTION WORKER ---
def emotion_worker(input_q, output_dict):
    """Runs in a separate process to avoid blocking the main UI."""
    while True:
        try:
            item = input_q.get()
            if item is None: break
            track_id, face_img = item
            # Use lightweight model for speed
            results = DeepFace.analyze(
                face_img, actions=['emotion'], 
                enforce_detection=False, silent=True
            )
            output_dict[track_id] = results[0]['dominant_emotion']
        except Exception:
            pass

# --- ANALYTICS UTILS ---
def get_angle(p1, p2):
    """Calculates angle between two points relative to the horizontal."""
    return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))

def is_facing(keypoints):
    """Estimates if user is facing camera based on eye/nose symmetry."""
    # COCO Keypoints: 0: nose, 1: l_eye, 2: r_eye
    nose, l_eye, r_eye = keypoints[0], keypoints[1], keypoints[2]
    if nose[2] < 0.5 or l_eye[2] < 0.5 or r_eye[2] < 0.5:
        return False
    # Check if nose is roughly between eyes
    dist_l = abs(nose[0] - l_eye[0])
    dist_r = abs(nose[0] - r_eye[0])
    ratio = min(dist_l, dist_r) / (max(dist_l, dist_r) + 1e-6)
    return ratio > 0.4

# --- MAIN ENGINE ---
class SurveillanceApp:
    def __init__(self):
        self.model = YOLO(YOLO_MODEL)
        self.vcap = VideoCaptureAsync(0)
        self.track_history = defaultdict(lambda: deque(maxlen=30))
        self.last_emotion_time = {}
        
        # Shared memory for multiprocessing
        self.manager = multiprocessing.Manager()
        self.emotion_results = self.manager.dict()
        self.emotion_queue = multiprocessing.Queue(maxsize=5)
        self.worker = multiprocessing.Process(
            target=emotion_worker, args=(self.emotion_queue, self.emotion_results)
        )
        self.worker.start()

    def run(self):
        print("System Running. Press 'q' to quit.")
        prev_time = time.time()
        
        while True:
            frame = self.vcap.read()
            if frame is None: continue
            
            # YOLO Tracking (with ByteTrack)
            results = self.model.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml")
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().numpy()
                keypoints = results[0].keypoints.data.cpu().numpy() # (N, 17, 3)

                for i, track_id in enumerate(track_ids):
                    kp = keypoints[i]
                    box = boxes[i].astype(int)
                    
                    # 1. Improved Fall Detection using Pose
                    # Angle between neck and hip
                    neck = (kp[5] + kp[6]) / 2 # Mid-shoulder
                    hip = (kp[11] + kp[12]) / 2 # Mid-hip
                    angle = abs(get_angle(neck, hip))
                    
                    # 2. Movement Analysis
                    self.track_history[track_id].append((neck[1], time.time()))
                    velocity = 0
                    if len(self.track_history[track_id]) > 1:
                        dy = self.track_history[track_id][-1][0] - self.track_history[track_id][0][0]
                        dt = self.track_history[track_id][-1][1] - self.track_history[track_id][0][1]
                        velocity = dy / dt if dt > 0 else 0

                    is_fallen = angle < FALL_ANGLE_THRESHOLD and velocity > FALL_VELOCITY_THRESHOLD
                    
                    # 3. Facing & Emotion Analysis
                    facing = is_facing(kp)
                    current_emotion = self.emotion_results.get(track_id, "analyzing...")
                    
                    # Trigger background emotion analysis if interval passed
                    now = time.time()
                    if now - self.last_emotion_time.get(track_id, 0) > EMOTION_CHECK_INTERVAL:
                        face_crop = frame[box[1]:box[3], box[0]:box[2]]
                        if face_crop.size > 0 and not self.emotion_queue.full():
                            self.emotion_queue.put((track_id, face_crop.copy()))
                            self.last_emotion_time[track_id] = now

                    # 4. Visualization
                    color = (0, 0, 255) if is_fallen else (0, 255, 0) if facing else (255, 255, 255)
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                    
                    label = f"ID:{track_id} {current_emotion}"
                    if is_fallen: label += " | FALL!"
                    cv2.putText(frame, label, (box[0], box[1]-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Performance Metrics
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.imshow("Advanced Surveillance", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        self.cleanup()

    def cleanup(self):
        self.vcap.stop()
        self.emotion_queue.put(None)
        self.worker.join()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Required for Windows multiprocessing
    multiprocessing.freeze_support()
    app = SurveillanceApp()
    app.run()