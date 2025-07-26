#
# VisionX - Main Application
#

import os
# FIX: This line must be at the very top to prevent a crash from library conflicts.
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import cv2
import time
import numpy as np
import faiss
import pickle
import datetime
import threading
import queue
from inference import FaceAnalysis
from database import init_db, log_exit_event

# --- Constants & Configuration ---
VIDEO_SOURCE = 0
RECOGNITION_THRESHOLD = 1.1
SESSION_TIMEOUT = 2.0
UNAUTHORIZED_DIRECTIONS = {'left', 'right', 'top'}

# --- Load Resources ---
print("Loading models and face index...")
providers = ['CUDAExecutionProvider','CPUExecutionProvider']
face_analyzer = FaceAnalysis(
    yolo_path="models/yolov8-face.onnx",
    arcface_path="models/arcface_w600k_r50.onnx",
    providers=providers
)
index = faiss.read_index("face_index.bin")
with open("labels.pkl", 'rb') as f:
    labels = pickle.load(f)
print("Resources loaded successfully.")

# --- Queues, Globals, and Helpers ---
frame_queue = queue.Queue(maxsize=1)
result_queue = queue.Queue(maxsize=1)
active_sessions = {}
frame_width, frame_height = 0, 0

def get_exit_direction(x1, y1, x2, y2, fw, fh):
    center_x = (x1 + x2) / 2
    if center_x < fw * 0.15: return "left"
    if center_x > fw * 0.85: return "right"
    if y1 < fh * 0.15: return "top"
    if y2 > fh * 0.85: return "bottom"
    return "unknown"

# --- Worker Thread for Processing ---
def face_processing_worker():
    global active_sessions, frame_width, frame_height
    while True:
        try:
            frame = frame_queue.get()
            face_boxes = face_analyzer.detect_faces(frame)
            results = []
            unknown_person_in_frame = False

            for box in face_boxes:
                x1, y1, x2, y2 = [max(0, val) for val in box]
                face_img = frame[y1:y2, x1:x2]
                if face_img.size == 0: continue

                name = "Unknown"
                embedding = face_analyzer.get_embedding(face_img)
                distances, indices = index.search(np.array([embedding], dtype=np.float32), k=1)
                if distances[0][0] < RECOGNITION_THRESHOLD:
                    name = labels[indices[0][0]]
                
                expression = face_analyzer.get_expression(face_img)
                age = face_analyzer.get_age(face_img)
                
                if name == "Unknown":
                    unknown_person_in_frame = True

                results.append({'box': box, 'name': name, 'expression': expression, 'age': age})

                if name != "Unknown":
                    if name not in active_sessions:
                        active_sessions[name] = {"entry_time": datetime.datetime.now(), "last_seen": time.time(), "last_pos": box}
                        print(f"EVENT: {name} entered.")
                    else:
                        active_sessions[name]["last_seen"] = time.time()
                        active_sessions[name]["last_pos"] = box
            
            exited_people = []
            for name, session_data in list(active_sessions.items()):
                if time.time() - session_data["last_seen"] > SESSION_TIMEOUT:
                    exit_time = datetime.datetime.now()
                    direction = get_exit_direction(*session_data["last_pos"], frame_width, frame_height)
                    log_exit_event(name, session_data["entry_time"], exit_time, direction)
                    
                    if direction in UNAUTHORIZED_DIRECTIONS:
                        print(f"SECURITY ALERT: {name.upper()} used UNAUTHORIZED exit via {direction.upper()} side.")

                    exited_people.append(name)
            
            for name in exited_people:
                if name in active_sessions: del active_sessions[name]
            
            if not result_queue.full():
                result_queue.put({'faces': results, 'alert': unknown_person_in_frame})

        except queue.Empty: continue
        except Exception as e: print(f"Error in worker thread: {e}")

# --- Main Application Thread ---
if __name__ == "__main__":
    init_db()
    worker_thread = threading.Thread(target=face_processing_worker, daemon=True)
    worker_thread.start()

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    processed_results = {'faces': [], 'alert': False}
    print("Starting real-time recognition... Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret: break

        if not frame_queue.full(): frame_queue.put(frame)
        try:
            processed_results = result_queue.get_nowait()
        except queue.Empty:
            pass

        if processed_results['alert']:
            alert_text = "ALERT: UNKNOWN PERSON DETECTED!"
            cv2.putText(frame, alert_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        for res in processed_results['faces']:
            x1, y1, x2, y2 = [max(0, int(v)) for v in res['box']]
            name, expression, age = res['name'], res['expression'], res['age']
            color = (219, 255, 75) if name != "Unknown" else (0, 0, 255)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 1)
            
            info_text = f"{expression}, {age}"
            cv2.putText(frame, info_text, (x1 + 5, y2 - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.6, color, 1)

        cv2.imshow("VisionX - Real-Time Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    print("Shutting down...")
    cap.release()
    cv2.destroyAllWindows()