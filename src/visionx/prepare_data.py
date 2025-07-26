# In prepare_data.py

import os
from data_utils import process_and_augment_dataset
from inference import FaceAnalysis

# --- Configuration ---
RAW_DATA_PATH = "Data"
PROCESSED_DATA_PATH = "Processed_Data"
MODELS_DIR = "models"
YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "yolov8-face.onnx")
ARCFACE_MODEL_PATH = os.path.join(MODELS_DIR, "arcface_w600k_r50.onnx")

if __name__ == "__main__":
    if not os.path.exists(YOLO_MODEL_PATH) or not os.path.exists(ARCFACE_MODEL_PATH):
        print("❌ Error: Model files not found.")
        print(f"Please download 'yolov8-face.onnx' and 'arcface_w600k_r50.onnx'")
        print(f"and place them in the '{MODELS_DIR}' directory.")
    else:
        face_analyzer = FaceAnalysis(YOLO_MODEL_PATH, ARCFACE_MODEL_PATH)
        process_and_augment_dataset(RAW_DATA_PATH, PROCESSED_DATA_PATH, face_analyzer)