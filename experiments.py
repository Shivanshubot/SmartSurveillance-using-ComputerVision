import csv
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


POSE_MODEL = "yolov8n-pose.pt"
OBJECT_MODEL = "yolov8n.pt"
LOG_FILE = "surveillance_log.csv"
ALERT_DIR = "alerts"

YOLO_IMGSZ = 640
YOLO_CONF = 0.35
YOLO_IOU = 0.45
OBJECT_DETECTION_INTERVAL = 2

FALL_ANGLE_THRESHOLD = 45
FALL_VELOCITY_THRESHOLD = 150
EMOTION_CHECK_INTERVAL = 2.0
HIGH_RISK_THRESHOLD = 6
ALERT_COOLDOWN_SECONDS = 2.5

PERSON_CLASS = 0
OBJECT_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    24: "backpack",
    26: "handbag",
    28: "suitcase",
    43: "knife",
}
DANGEROUS_OBJECT_CLASSES = {43}

HIGH_RISK_COLOR = (0, 0, 255)
LOOKING_COLOR = (0, 255, 0)
NORMAL_COLOR = (160, 160, 160)
TEXT_COLOR = (255, 255, 255)


os.makedirs(ALERT_DIR, exist_ok=True)


class VideoCaptureAsync:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
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
            if self.q.full():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get() if not self.q.empty() else None

    def stop(self):
        self.stopped = True
        self.cap.release()


def emotion_worker(input_q, output_dict):
    """Runs in a separate process so emotion inference does not block the UI."""
    while True:
        item = input_q.get()
        if item is None:
            break

        track_id, face_img = item
        try:
            results = DeepFace.analyze(
                face_img,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend="opencv",
                silent=True,
            )
            if isinstance(results, list):
                result = results[0] if results else {}
            else:
                result = results or {}
            output_dict[track_id] = result.get("dominant_emotion", "unknown")
        except Exception:
            output_dict[track_id] = "unknown"


def get_angle(p1, p2):
    """Calculates angle between two points relative to the horizontal."""
    return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))


def clamp_box(box, frame_shape):
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return x1, y1, x2, y2


def is_facing(keypoints):
    """Estimates if the user is facing the camera from facial keypoint symmetry."""
    nose, left_eye, right_eye = keypoints[0], keypoints[1], keypoints[2]
    if nose[2] < 0.5 or left_eye[2] < 0.5 or right_eye[2] < 0.5:
        return False
    dist_left = abs(nose[0] - left_eye[0])
    dist_right = abs(nose[0] - right_eye[0])
    ratio = min(dist_left, dist_right) / (max(dist_left, dist_right) + 1e-6)
    return ratio > 0.4


def is_looking_at_camera(face_box, person_wh):
    """Checks whether the detected face is centered enough to be looking at camera."""
    fx, fy, fw, fh = face_box
    person_w, person_h = person_wh
    face_center_x = fx + fw / 2
    face_center_y = fy + fh / 2
    rel_x = face_center_x / max(1, person_w)
    rel_y = face_center_y / max(1, person_h)
    centered = 0.25 < rel_x < 0.75 and 0.15 < rel_y < 0.6
    size_ok = fw > person_w * 0.22
    return centered and size_ok


def center_inside(expanded_box, inner_box):
    x1, y1, x2, y2 = inner_box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    ex1, ey1, ex2, ey2 = expanded_box
    return ex1 <= cx <= ex2 and ey1 <= cy <= ey2


def intersection_area(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0
    return (ix2 - ix1) * (iy2 - iy1)


class EventLogger:
    @staticmethod
    def log_high_risk(track_id, bbox, emotion, reasons, object_labels):
        file_exists = os.path.exists(LOG_FILE)
        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "track_id": track_id,
            "bbox": str(bbox),
            "emotion": emotion,
            "risk_level": "HIGH",
            "reasons": ", ".join(reasons),
            "objects": ", ".join(object_labels),
        }
        with open(LOG_FILE, "a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    @staticmethod
    def save_alert_frame(frame, track_id):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(ALERT_DIR, f"high_risk_{track_id}_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        return filename


class SurveillanceApp:
    def __init__(self, video_source=0):
        self.pose_model = YOLO(POSE_MODEL)
        self.object_model = YOLO(OBJECT_MODEL)
        self.vcap = VideoCaptureAsync(video_source)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        self.track_history = defaultdict(lambda: deque(maxlen=30))
        self.last_emotion_time = {}
        self.last_alert_time = {}
        self.frame_index = 0
        self.cached_objects = []

        self.manager = multiprocessing.Manager()
        self.emotion_results = self.manager.dict()
        self.emotion_queue = multiprocessing.Queue(maxsize=5)
        self.worker = multiprocessing.Process(
            target=emotion_worker,
            args=(self.emotion_queue, self.emotion_results),
            daemon=True,
        )
        self.worker.start()

    def detect_objects(self, frame):
        """Detects non-person objects and keeps only the categories we care about."""
        results = self.object_model.predict(
            frame,
            imgsz=YOLO_IMGSZ,
            conf=YOLO_CONF,
            iou=YOLO_IOU,
            verbose=False,
        )
        detections = []
        if not results:
            return detections

        result = results[0]
        if result.boxes is None:
            return detections

        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            class_id = int(cls)
            if class_id == PERSON_CLASS or class_id not in OBJECT_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, box.tolist())
            detections.append(
                {
                    "bbox": (x1, y1, x2, y2),
                    "class_id": class_id,
                    "label": OBJECT_CLASSES[class_id],
                    "confidence": float(conf),
                    "dangerous": class_id in DANGEROUS_OBJECT_CLASSES,
                }
            )
        return detections

    def detect_face(self, frame, person_box):
        """Detects a face inside a person box and determines camera attention."""
        x1, y1, x2, y2 = clamp_box(person_box, frame.shape)
        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            return None

        gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )
        if len(faces) == 0:
            return None

        fx, fy, fw, fh = max(faces, key=lambda face: face[2] * face[3])
        face_crop = person_crop[fy:fy + fh, fx:fx + fw]
        abs_box = (x1 + fx, y1 + fy, x1 + fx + fw, y1 + fy + fh)
        looking = is_looking_at_camera((fx, fy, fw, fh), (x2 - x1, y2 - y1))
        return {"bbox": abs_box, "crop": face_crop, "looking": looking}

    def objects_for_person(self, person_box, objects, frame_shape):
        """Associates nearby objects with a person box to avoid flagging everyone."""
        px1, py1, px2, py2 = person_box
        person_w = px2 - px1
        person_h = py2 - py1
        frame_h, frame_w = frame_shape[:2]

        margin_x = int(person_w * 0.2)
        margin_y = int(person_h * 0.1)
        expanded_box = (
            max(0, px1 - margin_x),
            max(0, py1 - margin_y),
            min(frame_w, px2 + margin_x),
            min(frame_h, py2 + margin_y),
        )

        matched_objects = []
        for obj in objects:
            if center_inside(expanded_box, obj["bbox"]) or intersection_area(person_box, obj["bbox"]) > 0:
                matched_objects.append(obj)
        return matched_objects

    def compute_risk(self, fallen, velocity, emotion, matched_objects):
        score = 0
        reasons = []

        if fallen:
            score += 6
            reasons.append("fall")
        if velocity > FALL_VELOCITY_THRESHOLD * 1.35:
            score += 2
            reasons.append("rapid movement")
        if emotion in {"angry", "fear"}:
            score += 2
            reasons.append(emotion)
        elif emotion == "disgust":
            score += 1
            reasons.append(emotion)
        if any(obj["dangerous"] for obj in matched_objects):
            score += 6
            reasons.append("dangerous object")

        return score, reasons

    def queue_emotion_analysis(self, track_id, face_crop):
        if face_crop is None or face_crop.size == 0:
            return
        if face_crop.shape[0] < 48 or face_crop.shape[1] < 48:
            return
        if self.emotion_queue.full():
            return

        face_img = cv2.resize(face_crop, (224, 224))
        self.emotion_queue.put((track_id, face_img.copy()))
        self.last_emotion_time[track_id] = time.time()

    def draw_label(self, frame, text, origin, color):
        x, y = origin
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(0, y - text_h - 8)
        cv2.rectangle(frame, (x, top), (x + text_w + 10, y), color, -1)
        cv2.putText(
            frame,
            text,
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            TEXT_COLOR,
            1,
        )

    def draw_person(self, frame, track_id, person_box, state):
        x1, y1, x2, y2 = person_box
        color = state["color"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2 if color == NORMAL_COLOR else 3)

        label_parts = [f"ID:{track_id}", state["emotion"]]
        if state["high_risk"]:
            label_parts.append("HIGH RISK")
        elif state["looking"]:
            label_parts.append("LOOKING")
        label = " | ".join(label_parts)
        self.draw_label(frame, label, (x1, max(20, y1)), color)

        if state["face_box"] is not None:
            fx1, fy1, fx2, fy2 = state["face_box"]
            face_color = LOOKING_COLOR if state["looking"] else NORMAL_COLOR
            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), face_color, 2)

        if state["reasons"]:
            reasons_text = ", ".join(state["reasons"][:2])
            cv2.putText(
                frame,
                reasons_text,
                (x1, min(frame.shape[0] - 10, y2 + 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

    def draw_objects(self, frame, objects):
        for obj in objects:
            x1, y1, x2, y2 = obj["bbox"]
            color = HIGH_RISK_COLOR if obj["dangerous"] else NORMAL_COLOR
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{obj['label']} {obj['confidence']:.2f}"
            self.draw_label(frame, label, (x1, max(20, y1)), color)

    def draw_stats(self, frame, fps, high_risk_count, looking_count, object_count):
        stats = [
            f"FPS: {fps:.1f}",
            f"High Risk: {high_risk_count}",
            f"Looking: {looking_count}",
            f"Objects: {object_count}",
        ]
        for index, stat in enumerate(stats):
            cv2.putText(
                frame,
                stat,
                (20, 35 + index * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                TEXT_COLOR,
                2,
            )

    def build_person_state(self, track_id, box, keypoints, frame, objects):
        person_box = clamp_box(box, frame.shape)
        x1, y1, x2, y2 = person_box

        neck = (keypoints[5] + keypoints[6]) / 2
        hip = (keypoints[11] + keypoints[12]) / 2
        pose_valid = (
            keypoints[5][2] > 0.3
            and keypoints[6][2] > 0.3
            and keypoints[11][2] > 0.3
            and keypoints[12][2] > 0.3
        )

        angle = abs(get_angle(neck, hip)) if pose_valid else 90
        now = time.time()
        self.track_history[track_id].append((float(neck[1]), now))

        velocity = 0.0
        if len(self.track_history[track_id]) > 1:
            dy = self.track_history[track_id][-1][0] - self.track_history[track_id][0][0]
            dt = self.track_history[track_id][-1][1] - self.track_history[track_id][0][1]
            velocity = dy / dt if dt > 0 else 0.0

        fallen = pose_valid and angle < FALL_ANGLE_THRESHOLD and velocity > FALL_VELOCITY_THRESHOLD

        face_info = self.detect_face(frame, person_box)
        looking = face_info["looking"] if face_info else is_facing(keypoints)

        current_emotion = self.emotion_results.get(track_id, "unknown")
        should_refresh_emotion = now - self.last_emotion_time.get(track_id, 0) > EMOTION_CHECK_INTERVAL
        if face_info and should_refresh_emotion:
            self.queue_emotion_analysis(track_id, face_info["crop"])

        matched_objects = self.objects_for_person(person_box, objects, frame.shape)
        risk_score, reasons = self.compute_risk(fallen, velocity, current_emotion, matched_objects)
        high_risk = risk_score >= HIGH_RISK_THRESHOLD

        if high_risk:
            color = HIGH_RISK_COLOR
        elif looking:
            color = LOOKING_COLOR
        else:
            color = NORMAL_COLOR

        return {
            "box": person_box,
            "face_box": face_info["bbox"] if face_info else None,
            "emotion": current_emotion,
            "looking": looking,
            "fallen": fallen,
            "high_risk": high_risk,
            "risk_score": risk_score,
            "reasons": reasons,
            "objects": matched_objects,
            "color": color,
        }

    def maybe_log_high_risk(self, frame, track_id, state):
        last_logged = self.last_alert_time.get(track_id, 0)
        if time.time() - last_logged < ALERT_COOLDOWN_SECONDS:
            return

        EventLogger.save_alert_frame(frame, track_id)
        EventLogger.log_high_risk(
            track_id=track_id,
            bbox=state["box"],
            emotion=state["emotion"],
            reasons=state["reasons"],
            object_labels=[obj["label"] for obj in state["objects"]],
        )
        self.last_alert_time[track_id] = time.time()

    def run(self):
        print("System Running. Press 'q' to quit.")
        previous_time = time.time()

        while True:
            frame = self.vcap.read()
            if frame is None:
                continue

            self.frame_index += 1

            pose_results = self.pose_model.track(
                frame,
                imgsz=YOLO_IMGSZ,
                conf=YOLO_CONF,
                iou=YOLO_IOU,
                persist=True,
                verbose=False,
                tracker="bytetrack.yaml",
            )

            if self.frame_index % OBJECT_DETECTION_INTERVAL == 0 or not self.cached_objects:
                self.cached_objects = self.detect_objects(frame)
            objects = self.cached_objects

            people_result = pose_results[0]
            high_risk_count = 0
            looking_count = 0

            if people_result.boxes is not None and people_result.boxes.id is not None:
                boxes = people_result.boxes.xyxy.cpu().numpy()
                track_ids = people_result.boxes.id.int().cpu().numpy()
                keypoints = people_result.keypoints.data.cpu().numpy()

                for index, track_id in enumerate(track_ids):
                    person_box = boxes[index].astype(int)
                    person_keypoints = keypoints[index]
                    state = self.build_person_state(
                        track_id=track_id,
                        box=person_box,
                        keypoints=person_keypoints,
                        frame=frame,
                        objects=objects,
                    )

                    if state["high_risk"]:
                        high_risk_count += 1
                        self.maybe_log_high_risk(frame, track_id, state)
                    if state["looking"] and not state["high_risk"]:
                        looking_count += 1

                    self.draw_person(frame, track_id, state["box"], state)

            self.draw_objects(frame, objects)

            current_time = time.time()
            fps = 1.0 / max(1e-6, current_time - previous_time)
            previous_time = current_time
            self.draw_stats(frame, fps, high_risk_count, looking_count, len(objects))

            cv2.imshow("Enhanced Surveillance - Face, Object and Risk View", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                filename = f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Saved {filename}")

        self.cleanup()

    def cleanup(self):
        self.vcap.stop()
        try:
            self.emotion_queue.put_nowait(None)
        except queue.Full:
            pass
        self.worker.join(timeout=2)
        if self.worker.is_alive():
            self.worker.terminate()
            self.worker.join(timeout=1)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    app = SurveillanceApp(video_source=0)
    app.run()
