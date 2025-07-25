# In app_dearpygui.py

import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import dearpygui.dearpygui as dpg
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
from src.visionx.database import init_db, log_exit_event

# --- Constants & Configuration ---
VIDEO_SOURCE = 0
RECOGNITION_THRESHOLD = 1.0
SESSION_TIMEOUT = 2.0
UNAUTHORIZED_DIRECTIONS = {'left', 'right', 'top'}
FRAME_WIDTH, FRAME_HEIGHT = 640, 480

# --- Queues, Globals, and Helpers ---
frame_queue = queue.Queue(maxsize=1)
result_queue = queue.Queue(maxsize=1)
active_sessions = {}
log_refresh_event = threading.Event()

def get_exit_direction(x1, y1, x2, y2, fw, fh):
    center_x = (x1 + x2) / 2
    if center_x < fw * 0.25: return "left"
    if center_x > fw * 0.75: return "right"
    if y1 < fh * 0.25: return "top"
    if y2 > fh * 0.75: return "bottom"
    return "unknown"

# --- Background Worker Thread ---
def face_processing_worker(face_analyzer, index, labels):
    global active_sessions
    while True:
        try:
            frame = frame_queue.get(timeout=1.0)
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
                if distances[0][0] < RECOGNITION_THRESHOLD: name = labels[indices[0][0]]
                
                expression = face_analyzer.get_expression(face_img)
                age = face_analyzer.get_age(face_img)
                if name == "Unknown": unknown_person_in_frame = True
                results.append({'box': box, 'name': name, 'expression': expression, 'age': age})

                if name != "Unknown":
                    if name not in active_sessions:
                        active_sessions[name] = {"entry_time": datetime.datetime.now(), "last_seen": time.time(), "last_pos": box}
                    else:
                        active_sessions[name]["last_seen"] = time.time()
            
            exited_people = []
            for name, data in list(active_sessions.items()):
                if time.time() - data["last_seen"] > SESSION_TIMEOUT:
                    direction = get_exit_direction(*data["last_pos"], FRAME_WIDTH, FRAME_HEIGHT)
                    log_exit_event(name, data["entry_time"], datetime.datetime.now(), direction)
                    log_refresh_event.set() # Signal the main thread
                    if direction in UNAUTHORIZED_DIRECTIONS: print(f"SECURITY ALERT: {name.upper()} used UNAUTHORIZED exit via {direction.upper()} side.")
                    exited_people.append(name)
            for name in exited_people:
                if name in active_sessions: del active_sessions[name]
            
            if not result_queue.full():
                result_queue.put({'faces': results, 'alert': unknown_person_in_frame})

        except queue.Empty: continue
        except Exception as e: print(f"Error in worker thread: {e}")

# --- Dear PyGui Main Application ---
def main():
    print("Loading models and face index...")
    providers = ['CUDAExecutionProvider','CPUExecutionProvider']
    face_analyzer = FaceAnalysis(yolo_path="models/yolov8-face.onnx", arcface_path="models/arcface_w600k_r50.onnx", providers=providers)
    index = faiss.read_index("face_index.bin")
    with open("labels.pkl", 'rb') as f:
        labels = pickle.load(f)
    init_db()
    print("Resources loaded successfully.")

    worker_thread = threading.Thread(target=face_processing_worker, args=(face_analyzer, index, labels), daemon=True)
    worker_thread.start()

    dpg.create_context()
    dpg.create_viewport(title='VisionX Dashboard')
    dpg.setup_dearpygui()

    with dpg.texture_registry():
        dpg.add_raw_texture(width=FRAME_WIDTH, height=FRAME_HEIGHT, default_value=np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 4), dtype=np.float32), format=dpg.mvFormat_Float_rgba, tag="video_texture")

    def refresh_logs():
        try:
            # Clear existing table data (but not headers)
            dpg.delete_item("log_table", children_only=True, slot=1)
            
            conn = sqlite3.connect("visionx_log.db")
            df = pd.read_sql_query("SELECT id, person_name, entry_time, exit_time, exit_direction FROM session_logs ORDER BY exit_time DESC", conn)
            conn.close()
            
            # Add new data rows to the existing table
            for i in range(df.shape[0]):
                with dpg.table_row(parent="log_table"):
                    for item in df.iloc[i]:
                        dpg.add_text(str(item))
        except Exception as e:
            print(f"Error refreshing logs: {e}")

    # FINAL LAYOUT FIX: A simplified and explicit structure
    with dpg.window(tag="main_window"):
        with dpg.group(horizontal=True):
            # Left Pane for Video
            with dpg.group():
                dpg.add_text("Live Camera Feed")
                dpg.add_image("video_texture")
            
            # Right Pane for Logs
            with dpg.group():
                dpg.add_text("Session Logs")
                dpg.add_button(label="Refresh Logs", callback=refresh_logs)
                with dpg.table(header_row=True, resizable=True, policy=dpg.mvTable_SizingStretchProp,
                               tag="log_table"):
                    dpg.add_table_column(label="ID", width_fixed=True, init_width_or_weight=40)
                    dpg.add_table_column(label="Name")
                    dpg.add_table_column(label="Entry Time")
                    dpg.add_table_column(label="Exit Time")
                    dpg.add_table_column(label="Exit Direction")

    dpg.show_viewport()
    dpg.set_primary_window("main_window", True)
    dpg.maximize_viewport()
    refresh_logs()
    
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    processed_results = {'faces': [], 'alert': False}

    while dpg.is_dearpygui_running():
        if log_refresh_event.is_set():
            refresh_logs()
            log_refresh_event.clear()

        ret, frame = cap.read()
        if not ret: 
            dpg.render_dearpygui_frame()
            continue
        
        frame_for_processing = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        
        if not frame_queue.full():
            frame_queue.put(frame_for_processing)
        
        try:
            processed_results = result_queue.get_nowait()
        except queue.Empty:
            pass

        if processed_results['alert']:
            cv2.putText(frame_for_processing, "ALERT: UNKNOWN PERSON!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        for res in processed_results['faces']:
            x1, y1, x2, y2 = [max(0, int(v)) for v in res['box']]
            name, expression, age = res['name'], res['expression'], res['age']
            color = (0, 250, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame_for_processing, (x1, y1), (x2, y2), color, 1)
            cv2.putText(frame_for_processing, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 1)
            
            cv2.putText(frame_for_processing, expression, (x1 + 5, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            cv2.putText(frame_for_processing, age, (x1 + 5, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        frame_rgba = cv2.cvtColor(frame_for_processing, cv2.COLOR_BGR2RGBA)
        frame_float = (frame_rgba.astype(np.float32) / 255.0)
        dpg.set_value("video_texture", frame_float)
        
        dpg.render_dearpygui_frame()

    cap.release()
    dpg.destroy_context()

if __name__ == "__main__":
    main()