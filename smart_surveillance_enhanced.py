import cv2
import numpy as np
import time
import os
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
from deepface import DeepFace
from dotenv import load_dotenv
from collections import deque, defaultdict
from threading import Thread, Lock
import queue

# ==================== CONFIGURATION ====================
load_dotenv()

class Config:
    # Paths
    ALERT_DIR = os.getenv("ALERT_SAVE_DIR", "alerts")
    LOG_CSV = os.getenv("LOG_CSV", "events_log.csv")
    
    # YOLO Settings
    YOLO_MODEL = "yolov8n.pt"
    YOLO_IMGSZ = 640
    YOLO_CONF = 0.35
    YOLO_IOU = 0.45
    
    # Processing
    PROCESS_EVERY_N_FRAMES = 2  # Process 1 in N frames
    EMOTION_EVERY_N_FRAMES = 6  # Emotion detection every N frames
    MAX_QUEUE_SIZE = 2
    
    # Detection Classes (COCO dataset)
    PERSON_CLASS = 0
    WEAPON_CLASSES = [43]  # knife
    BAG_CLASSES = [24, 26, 28]  # backpack, handbag, suitcase
    VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
    
    # Behavior Thresholds
    SPEED_RUNNING = 200
    SPEED_FAST_WALK = 100
    LOITER_RADIUS = 50
    LOITER_TIME = 10
    FALL_Y_THRESH = 70
    FALL_H_RATIO = 0.7
    CROWD_DENSITY_HIGH = 5
    
    # Risk Scoring
    RISK_HIGH = 7
    RISK_MEDIUM = 4
    
    # Tracking
    TRACK_MAX_AGE = 30
    EMOTION_HISTORY_SIZE = 5
    
    # Zones (normalized coordinates 0-1)
    RESTRICTED_ZONES = [
        # Example: [(x1, y1, x2, y2), ...]
        # (0.7, 0.0, 1.0, 0.3),  # Top-right corner
    ]

os.makedirs(Config.ALERT_DIR, exist_ok=True)

# ==================== THREADED VIDEO CAPTURE ====================
class VideoStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.lock = Lock()
        
    def start(self):
        Thread(target=self.update, daemon=True).start()
        return self
        
    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret, self.frame = ret, frame
                
    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None
            
    def stop(self):
        self.stopped = True
        self.cap.release()

# ==================== ENHANCED TRACKER ====================
class PersonTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 0
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        
    def update(self, yolo_results, frame):
        """Update tracks using YOLO's built-in tracking"""
        current_tracks = {}
        
        for result in yolo_results:
            if result.boxes.id is None:
                continue
                
            for box, track_id, cls in zip(result.boxes.xyxy, result.boxes.id, result.boxes.cls):
                if int(cls) != Config.PERSON_CLASS:
                    continue
                    
                tid = int(track_id)
                x1, y1, x2, y2 = map(int, box)
                bbox = (x1, y1, x2 - x1, y2 - y1)
                
                if tid not in self.tracks:
                    self.tracks[tid] = {
                        'id': tid,
                        'bbox': bbox,
                        'history': deque(maxlen=60),
                        'emotions': deque(maxlen=Config.EMOTION_HISTORY_SIZE),
                        'last_emotion_check': 0,
                        'frames_since_seen': 0,
                        'first_seen': time.time(),
                        'looking': False,
                        'fall_detected': False,
                        'zone_violation': False
                    }
                
                track = self.tracks[tid]
                track['bbox'] = bbox
                track['history'].append((time.time(), (x1 + (x2-x1)//2, y1 + (y2-y1)//2)))
                track['frames_since_seen'] = 0
                current_tracks[tid] = track
        
        # Age out old tracks
        for tid in list(self.tracks.keys()):
            if tid not in current_tracks:
                self.tracks[tid]['frames_since_seen'] += 1
                if self.tracks[tid]['frames_since_seen'] > Config.TRACK_MAX_AGE:
                    del self.tracks[tid]
        
        return self.tracks
    
    def analyze_person(self, track, frame, frame_count):
        """Analyze person: emotion, attention, behavior"""
        x, y, w, h = track['bbox']
        
        # Compute speed
        speed = self._compute_speed(track['history'])
        
        # Check fall
        fall = self._detect_fall(track['history'])
        if fall:
            track['fall_detected'] = True
        
        # Check loitering
        loiter = self._check_loitering(track['history'], track['first_seen'])
        
        # Check zone violation
        zone_violation = self._check_zone_violation(x, y, w, h, frame.shape)
        track['zone_violation'] = zone_violation
        
        # Emotion & attention detection (throttled)
        if frame_count - track['last_emotion_check'] >= Config.EMOTION_EVERY_N_FRAMES:
            emotion, looking = self._analyze_face(frame, x, y, w, h)
            if emotion != "unknown":
                track['emotions'].append(emotion)
            track['looking'] = looking
            track['last_emotion_check'] = frame_count
        
        # Get dominant emotion
        dominant_emotion = self._get_dominant_emotion(track['emotions'])
        
        return {
            'speed': speed,
            'fall': fall,
            'loiter': loiter,
            'emotion': dominant_emotion,
            'looking': track['looking'],
            'zone_violation': zone_violation
        }
    
    def _analyze_face(self, frame, x, y, w, h):
        """Detect face, emotion, and attention"""
        crop = frame[y:y+h, x:x+w]
        if crop.size == 0:
            return "unknown", False
        
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        
        if len(faces) == 0:
            return "unknown", False
        
        fx, fy, fw, fh = faces[0]
        
        # Check attention
        looking = self._is_looking_at_camera((fx, fy, fw, fh), (w, h))
        
        # Emotion detection
        emotion = "unknown"
        try:
            face_crop = crop[fy:fy+fh, fx:fx+fw]
            if face_crop.shape[0] > 48 and face_crop.shape[1] > 48:
                face_crop = cv2.resize(face_crop, (224, 224))
                result = DeepFace.analyze(face_crop, actions=["emotion"],
                                         enforce_detection=False, 
                                         detector_backend="opencv",
                                         silent=True)
                emotion = result[0]["dominant_emotion"]
        except:
            pass
        
        return emotion, looking
    
    def _is_looking_at_camera(self, face_box, person_wh):
        """Check if person is looking at camera"""
        fx, fy, fw, fh = face_box
        pw, ph = person_wh
        cx, cy = fx + fw/2, fy + fh/2
        rx, ry = cx / pw, cy / ph
        centered = 0.25 < rx < 0.75 and 0.15 < ry < 0.6
        size_ok = fw > pw * 0.22
        return centered and size_ok
    
    def _compute_speed(self, history, window=1.0):
        """Compute movement speed"""
        if len(history) < 2:
            return 0
        now = time.time()
        recent = [h for h in history if now - h[0] <= window]
        if len(recent) < 2:
            recent = list(history)[-2:]
        if len(recent) < 2:
            return 0
        (t1, p1), (t2, p2) = recent[0], recent[-1]
        dist = np.linalg.norm(np.array(p2) - np.array(p1))
        dt = max(0.001, t2 - t1)
        return dist / dt
    
    def _detect_fall(self, history):
        """Detect sudden fall"""
        if len(history) < 3:
            return False
        t1, (x1, y1) = history[-2]
        t2, (x2, y2) = history[-1]
        return (y2 - y1) > Config.FALL_Y_THRESH
    
    def _check_loitering(self, history, first_seen):
        """Check if person is loitering"""
        if len(history) < 10:
            return False
        duration = time.time() - first_seen
        if duration < Config.LOITER_TIME:
            return False
        recent = list(history)[-20:]
        xs = [p[0] for _, p in recent]
        ys = [p[1] for _, p in recent]
        range_x = max(xs) - min(xs)
        range_y = max(ys) - min(ys)
        return range_x < Config.LOITER_RADIUS and range_y < Config.LOITER_RADIUS
    
    def _check_zone_violation(self, x, y, w, h, frame_shape):
        """Check if person is in restricted zone"""
        fh, fw = frame_shape[:2]
        cx, cy = (x + w/2) / fw, (y + h/2) / fh
        
        for zx1, zy1, zx2, zy2 in Config.RESTRICTED_ZONES:
            if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
                return True
        return False
    
    def _get_dominant_emotion(self, emotions):
        """Get most common recent emotion"""
        if not emotions:
            return "neutral"
        return max(set(emotions), key=list(emotions).count)

# ==================== OBJECT DETECTOR ====================
class ObjectDetector:
    def __init__(self):
        self.detected_objects = defaultdict(list)
        
    def detect(self, yolo_results):
        """Detect weapons, bags, vehicles"""
        self.detected_objects.clear()
        
        for result in yolo_results:
            for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                cls_id = int(cls)
                
                if cls_id in Config.WEAPON_CLASSES:
                    self.detected_objects['weapons'].append(box)
                elif cls_id in Config.BAG_CLASSES:
                    self.detected_objects['bags'].append(box)
                elif cls_id in Config.VEHICLE_CLASSES:
                    self.detected_objects['vehicles'].append(box)
        
        return self.detected_objects

# ==================== RISK ANALYZER ====================
class RiskAnalyzer:
    @staticmethod
    def compute_risk(analysis, objects):
        """Compute risk score and level"""
        score = 0
        reasons = []
        
        # Emotion-based risk
        emotion = analysis['emotion']
        if emotion in ["angry", "fear"]:
            score += 3
            reasons.append(f"{emotion}")
        elif emotion == "disgust":
            score += 2
            reasons.append("disgust")
        
        # Speed-based risk
        if analysis['speed'] > Config.SPEED_RUNNING:
            score += 4
            reasons.append("running")
        elif analysis['speed'] > Config.SPEED_FAST_WALK:
            score += 2
            reasons.append("fast movement")
        
        # Fall detection
        if analysis['fall']:
            score += 5
            reasons.append("fall detected")
        
        # Loitering
        if analysis['loiter']:
            score += 3
            reasons.append("loitering")
        
        # Zone violation
        if analysis['zone_violation']:
            score += 4
            reasons.append("restricted zone")
        
        # Object detection
        if objects['weapons']:
            score += 10
            reasons.append("WEAPON")
        if len(objects['bags']) > 1:
            score += 2
            reasons.append("multiple bags")
        
        # Determine level
        if score >= Config.RISK_HIGH:
            level = "HIGH"
        elif score >= Config.RISK_MEDIUM:
            level = "MEDIUM"
        else:
            level = "LOW"
        
        return score, level, reasons

# ==================== LOGGER ====================
class EventLogger:
    @staticmethod
    def log_event(event_type, track_id, bbox, emotion, risk_level, details=""):
        """Log event to CSV"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = {
            "timestamp": timestamp,
            "event_type": event_type,
            "track_id": track_id,
            "bbox": str(bbox),
            "emotion": emotion,
            "risk_level": risk_level,
            "details": details
        }
        df = pd.DataFrame([row])
        header = not os.path.exists(Config.LOG_CSV)
        df.to_csv(Config.LOG_CSV, mode="a", header=header, index=False)
    
    @staticmethod
    def save_alert_image(frame, track_id, risk_level):
        """Save alert image"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{Config.ALERT_DIR}/alert_{risk_level}_{track_id}_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        return filename

# ==================== VISUALIZATION ====================
class Visualizer:
    def __init__(self, frame_shape):
        self.frame_shape = frame_shape
        self.colors = {
            'HIGH': (0, 0, 255),      # Red
            'MEDIUM': (0, 165, 255),  # Orange
            'LOW': (0, 255, 0),       # Green
            'LOOKING': (0, 255, 0),   # Green
            'NEUTRAL': (200, 200, 200) # Gray
        }
    
    def draw_track(self, frame, track, analysis, risk_level):
        """Draw bounding box and info for a track"""
        x, y, w, h = track['bbox']
        
        # Determine color
        if risk_level == "HIGH":
            color = self.colors['HIGH']
        elif risk_level == "MEDIUM":
            color = self.colors['MEDIUM']
        elif analysis['looking']:
            color = self.colors['LOOKING']
        else:
            color = self.colors['NEUTRAL']
        
        # Draw box
        thickness = 3 if risk_level != "LOW" else 2
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
        
        # Draw label
        if analysis['looking'] or risk_level != "LOW":
            label = f"ID:{track['id']} {analysis['emotion']}"
            if risk_level != "LOW":
                label += f" | {risk_level}"
            
            # Background for text
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x, y-th-8), (x+tw+10, y), color, -1)
            cv2.putText(frame, label, (x+5, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw indicators
        if analysis['speed'] > Config.SPEED_RUNNING:
            cv2.putText(frame, "RUNNING", (x, y+h+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if analysis['fall']:
            cv2.putText(frame, "FALL!", (x, y-30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    def draw_objects(self, frame, objects):
        """Draw detected objects"""
        for weapon_box in objects['weapons']:
            x1, y1, x2, y2 = map(int, weapon_box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, "WEAPON!", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    def draw_zones(self, frame):
        """Draw restricted zones"""
        fh, fw = frame.shape[:2]
        for zx1, zy1, zx2, zy2 in Config.RESTRICTED_ZONES:
            x1, y1 = int(zx1 * fw), int(zy1 * fh)
            x2, y2 = int(zx2 * fw), int(zy2 * fh)
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
            cv2.putText(frame, "RESTRICTED", (x1+5, y1+25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    def draw_stats(self, frame, stats):
        """Draw statistics panel"""
        panel_h = 120
        panel = np.zeros((panel_h, frame.shape[1], 3), dtype=np.uint8)
        
        y_pos = 25
        cv2.putText(panel, f"FPS: {stats['fps']:.1f}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.putText(panel, f"Tracked: {stats['tracked']}", (150, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.putText(panel, f"High Risk: {stats['high_risk']}", (300, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.putText(panel, f"Alerts: {stats['alerts']}", (480, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        y_pos = 60
        cv2.putText(panel, f"Weapons: {stats['weapons']}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        cv2.putText(panel, f"Attention: {stats['looking']}", (150, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.putText(panel, f"Crowd: {stats['crowd']}", (300, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return np.vstack([frame, panel])

# ==================== MAIN SYSTEM ====================
class SmartSurveillance:
    def __init__(self):
        print("Initializing Smart Surveillance System...")
        self.yolo = YOLO(Config.YOLO_MODEL)
        self.tracker = PersonTracker()
        self.detector = ObjectDetector()
        self.logger = EventLogger()
        
        self.frame_count = 0
        self.fps = 0
        self.alert_count = 0
        self.fps_counter = deque(maxlen=30)
        
        print("System ready!")
    
    def run(self, video_source=0):
        """Main processing loop"""
        stream = VideoStream(video_source).start()
        time.sleep(1.0)
        
        last_time = time.time()
        
        while True:
            ret, frame = stream.read()
            if not ret:
                break
            
            self.frame_count += 1
            
            # Process every N frames for performance
            if self.frame_count % Config.PROCESS_EVERY_N_FRAMES != 0:
                continue
            
            # Run YOLO detection with tracking
            results = self.yolo.track(frame, 
                                     imgsz=Config.YOLO_IMGSZ,
                                     conf=Config.YOLO_CONF,
                                     iou=Config.YOLO_IOU,
                                     persist=True,
                                     verbose=False,
                                     tracker="bytetrack.yaml")
            
            # Update tracks
            tracks = self.tracker.update(results, frame)
            
            # Detect objects
            objects = self.detector.detect(results)
            
            # Initialize visualizer
            if self.frame_count == Config.PROCESS_EVERY_N_FRAMES:
                self.visualizer = Visualizer(frame.shape)
            
            # Draw zones
            self.visualizer.draw_zones(frame)
            
            # Analyze and visualize each track
            high_risk_count = 0
            looking_count = 0
            
            for track_id, track in tracks.items():
                # Analyze person
                analysis = self.tracker.analyze_person(track, frame, self.frame_count)
                
                # Compute risk
                risk_score, risk_level, reasons = RiskAnalyzer.compute_risk(analysis, objects)
                
                # Count stats
                if risk_level == "HIGH":
                    high_risk_count += 1
                if analysis['looking']:
                    looking_count += 1
                
                # Visualize
                self.visualizer.draw_track(frame, track, analysis, risk_level)
                
                # Handle high risk
                if risk_level == "HIGH":
                    self.alert_count += 1
                    filename = self.logger.save_alert_image(frame, track_id, risk_level)
                    self.logger.log_event("HIGH_RISK", track_id, track['bbox'],
                                         analysis['emotion'], risk_level,
                                         ", ".join(reasons))
            
            # Draw detected objects
            self.visualizer.draw_objects(frame, objects)
            
            # Calculate FPS
            current_time = time.time()
            self.fps_counter.append(1.0 / max(0.001, current_time - last_time))
            self.fps = np.mean(self.fps_counter)
            last_time = current_time
            
            # Crowd density
            crowd_level = "HIGH" if len(tracks) >= Config.CROWD_DENSITY_HIGH else "NORMAL"
            
            # Draw stats
            stats = {
                'fps': self.fps,
                'tracked': len(tracks),
                'high_risk': high_risk_count,
                'alerts': self.alert_count,
                'weapons': len(objects['weapons']),
                'looking': looking_count,
                'crowd': crowd_level
            }
            frame = self.visualizer.draw_stats(frame, stats)
            
            # Display
            cv2.imshow("Enhanced Smart Surveillance System", frame)
            
            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(f"screenshot_{int(time.time())}.jpg", frame)
                print("Screenshot saved!")
        
        stream.stop()
        cv2.destroyAllWindows()
        print(f"\nSession complete. Total alerts: {self.alert_count}")

# ==================== ENTRY POINT ====================
if __name__ == "__main__":
    system = SmartSurveillance()
    system.run(video_source=0)
