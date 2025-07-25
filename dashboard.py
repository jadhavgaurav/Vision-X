# In dashboard.py

import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import cv2
import time
import numpy as np
import faiss
import pickle
import datetime
import threading
import queue
import pandas as pd
import sqlite3
from src.visionx.inference import FaceAnalysis
from src.visionx.database import log_exit_event

# --- Page Configuration ---
st.set_page_config(page_title="VisionX Dashboard", layout="wide")
st.title("VisionX - Real-Time Face Recognition System")

# --- Resource Loading (Cached) ---
@st.cache_resource
def load_resources():
    print("Loading resources...")
    providers = ['CPUExecutionProvider']
    face_analyzer = FaceAnalysis(
        yolo_path="models/yolov8-face.onnx",
        arcface_path="models/arcface_w600k_r50.onnx",
        providers=providers
    )
    index = faiss.read_index("face_index.bin")
    with open("labels.pkl", 'rb') as f:
        labels = pickle.load(f)
    print("Resources loaded successfully.")
    return face_analyzer, index, labels

face_analyzer, index, labels = load_resources()

# --- Session State & Global State ---
if 'active_sessions' not in st.session_state:
    st.session_state.active_sessions = {}
if 'logs_df' not in st.session_state:
    st.session_state.logs_df = pd.DataFrame()
if 'latest_results' not in st.session_state:
    st.session_state.latest_results = {'faces': [], 'alert': False}
if 'worker_thread' not in st.session_state:
    st.session_state.worker_thread = None

frame_queue = queue.Queue(maxsize=2)
lock = threading.Lock()

# --- Constants & Helpers ---
RECOGNITION_THRESHOLD = 1.1
SESSION_TIMEOUT = 3.0
UNAUTHORIZED_DIRECTIONS = {'left', 'right', 'top'}

def get_exit_direction(x1, y1, x2, y2, fw, fh):
    center_x = (x1 + x2) / 2
    if center_x < fw * 0.15: return "left"
    if center_x > fw * 0.85: return "right"
    if y1 < fh * 0.15: return "top"
    if y2 > fh * 0.85: return "bottom"
    return "unknown"

# --- Background Worker Thread ---
def face_processing_worker():
    while True:
        try:
            frame, frame_time = frame_queue.get(timeout=1)
            frame_height, frame_width, _ = frame.shape
            
            face_boxes = face_analyzer.detect_faces(frame)
            results_list = []
            unknown_person_in_frame = False

            for box in face_boxes:
                x1, y1, x2, y2 = [max(0, val) for val in box]
                face_img = frame[y1:y2, x1:x2]
                if face_img.size == 0: continue

                name = "Unknown"
                embedding = face_analyzer.get_embedding(face_img)
                distances, indices = index.search(np.array([embedding], dtype=np.float32), k=1)
                if distances[0][0] < RECOGNITION_THRESHOLD: name = labels[indices[0][0]]
                
                expression = face_analyzer.get_expression(face_img)
                age = face_analyzer.get_age(face_img)
                
                if name == "Unknown": unknown_person_in_frame = True
                results_list.append({'box': box, 'name': name, 'expression': expression, 'age': age})

                with lock:
                    if name != "Unknown":
                        if name not in st.session_state.active_sessions:
                            st.session_state.active_sessions[name] = {"entry_time": datetime.datetime.now(), "last_seen": frame_time, "last_pos": box}
                        else:
                            st.session_state.active_sessions[name]["last_seen"] = frame_time
                            st.session_state.active_sessions[name]["last_pos"] = box
            
            with lock:
                exited_people = []
                for name, data in st.session_state.active_sessions.items():
                    if frame_time - data["last_seen"] > SESSION_TIMEOUT:
                        exited_people.append(name)
                
                for name in exited_people:
                    data = st.session_state.active_sessions.pop(name)
                    direction = get_exit_direction(*data["last_pos"], frame_width, frame_height)
                    log_exit_event(name, data["entry_time"], datetime.datetime.now(), direction)
                    if direction in UNAUTHORIZED_DIRECTIONS: print(f"SECURITY ALERT: {name} used unauthorized exit: {direction}")

            st.session_state.latest_results = {'faces': results_list, 'alert': unknown_person_in_frame}

        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in worker thread: {e}")

# --- Streamlit UI Components ---
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Live Camera Feed")
    
    # UPDATED: We refactor the processor to explicitly receive the queue
    class VisionXProcessor:
        def __init__(self, frame_queue: queue.Queue):
            self.frame_queue = frame_queue

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            
            try:
                # Use the queue passed during initialization
                self.frame_queue.put_nowait((img, time.time()))
            except queue.Full:
                pass

            results = st.session_state.get('latest_results', {'faces': [], 'alert': False})
            
            if results and results['faces']:
                if results['alert']:
                    cv2.putText(img, "ALERT: UNKNOWN PERSON!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                
                for res in results['faces']:
                    x1, y1, x2, y2 = [max(0, int(v)) for v in res['box']]
                    name, expression, age = res['name'], res['expression'], res['age']
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    info_text = f"{expression}, {age}"
                    cv2.putText(img, info_text, (x1 + 5, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="visionx-stream",
        # UPDATED: We use a lambda function to pass the queue to the processor instance
        video_processor_factory=lambda: VisionXProcessor(frame_queue),
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.header("Session Logs")
    if st.button("Refresh Logs"):
        try:
            conn = sqlite3.connect("visionx_log.db")
            st.session_state.logs_df = pd.read_sql_query("SELECT * FROM session_logs ORDER BY exit_time DESC", conn)
            conn.close()
        except Exception as e: st.error(f"Could not load logs: {e}")
            
    st.dataframe(st.session_state.logs_df, use_container_width=True)

# Start the worker thread only once
if st.session_state.worker_thread is None:
    st.session_state.worker_thread = threading.Thread(target=face_processing_worker, daemon=True)
    st.session_state.worker_thread.start()