#
# VisionX - Main Application
# This script runs the real-time face recognition and expression analysis.
#

import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import cv2
import time
import numpy as np
import faiss
import pickle
import datetime
import threading
import queue
from src.visionx.inference import FaceAnalysis
from src.visionx.database import init_db, log_exit_event

# --- Constants & Configuration ---
VIDEO_SOURCE = 0  # 0 for webcam, or a path to a video file
RECOGNITION_THRESHOLD = 1.1  # L2 distance. Lower is stricter. Tune for your environment.
SESSION_TIMEOUT = 2.0  # Seconds of absence before a person is considered 'exited'.

# --- Load Resources ---
print("Loading models and face index...")
# Define providers. Forcing CPU to avoid system-level CUDA setup issues.
providers = ['CUDAExecutionProvider','CPUExecutionProvider']

# Initialize the FaceAnalysis class using the Transformers pipeline for expressions and age
face_analyzer = FaceAnalysis(
    yolo_path="models/yolov8-face.onnx", 
    arcface_path="models/arcface_w600k_r50.onnx",
    providers=providers
)
index = faiss.read_index("face_index.bin")
with open("labels.pkl", 'rb') as f:
    labels = pickle.load(f)
print("Resources loaded successfully.")

# --- Queues for Thread-Safe Communication ---
frame_queue = queue.Queue(maxsize=1)
result_queue = queue.Queue(maxsize=1)

# --- Global variables for session management ---
active_sessions = {}
frame_width = 0
frame_height = 0

def get_exit_direction(x1, y1, x2, y2, fw, fh):
    """Determines exit direction based on the last known bounding box position."""
    center_x = (x1 + x2) / 2
    if center_x < fw * 0.15: return "left"
    if center_x > fw * 0.85: return "right"
    if y1 < fh * 0.15: return "top"
    if y2 > fh * 0.85: return "bottom"
    return "unknown"

def face_processing_worker():
    """
    This function runs in a separate thread.
    It gets frames from a queue, processes them, and puts the results in another queue.
    """
    global active_sessions, frame_width, frame_height
    while True:
        try:
            frame = frame_queue.get()
            
            face_boxes = face_analyzer.detect_faces(frame)
            results = []
            
            for box in face_boxes:
                x1, y1, x2, y2 = [max(0, val) for val in box]
                face_img = frame[y1:y2, x1:x2]
                
                if face_img.size == 0:
                    continue

                # Get recognition embedding and find name
                name = "Unknown"
                embedding = face_analyzer.get_embedding(face_img)
                embedding_np = np.array([embedding], dtype=np.float32)
                distances, indices = index.search(embedding_np, k=1)
                if distances[0][0] < RECOGNITION_THRESHOLD:
                    name = labels[indices[0][0]]
                
                # Get facial expression
                expression = face_analyzer.get_expression(face_img)

                # Get age prediction
                age = face_analyzer.get_age(face_img)
                
                # results dictionary to include age
                results.append({'box': box, 'name': name, 'expression': expression, 'age': age})

                # Manage Session Entry/Update
                if name != "Unknown":
                    if name not in active_sessions:
                        active_sessions[name] = {"entry_time": datetime.datetime.now(), "last_seen": time.time(), "last_pos": box}
                        print(f"EVENT: {name} entered.")
                    else:
                        active_sessions[name]["last_seen"] = time.time()
                        active_sessions[name]["last_pos"] = box
            
            # Manage Session Exits
            exited_people = []
            for name, session_data in list(active_sessions.items()):
                if time.time() - session_data["last_seen"] > SESSION_TIMEOUT:
                    exit_time = datetime.datetime.now()
                    direction = get_exit_direction(*session_data["last_pos"], frame_width, frame_height)
                    log_exit_event(name, session_data["entry_time"], exit_time, direction)
                    exited_people.append(name)
            
            for name in exited_people:
                if name in active_sessions: del active_sessions[name]

            # Send results to the main thread
            if not result_queue.full():
                result_queue.put(results)

        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in worker thread: {e}")


# --- Main Application Thread ---
if __name__ == "__main__":
    init_db()

    # Start the background processing thread
    worker_thread = threading.Thread(target=face_processing_worker, daemon=True)
    worker_thread.start()

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        raise IOError("Cannot open video source")
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    processed_results = []
    print("Starting real-time recognition... Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Put the latest frame into the queue for the worker, but don't wait
        if not frame_queue.full():
            frame_queue.put(frame)

        # Get the latest processed results if available, otherwise use old ones
        try:
            processed_results = result_queue.get_nowait()
        except queue.Empty:
            pass

        # Draw the latest results on the current frame
        for res in processed_results:
            x1, y1, x2, y2 = [max(0, int(v)) for v in res['box']]
            # COMPLETE: Unpack the age from the results
            name, expression, age = res['name'], res['expression'], res['age']
            
            color = (219, 255, 75) if name != "Unknown" else (0, 0, 255)
            
            # Draw bounding box and name
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 1)
            

            # COMPLETE: Display expression and age inside the box
            info_text = f"{expression},  {age}"
            cv2.putText(frame, info_text, (x1 + 5, y2 - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.6, color, 1)

        cv2.imshow("VisionX - Real-Time Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Shutting down...")
    cap.release()
    cv2.destroyAllWindows()