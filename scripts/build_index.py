# In build_index.py

import os
import cv2
import numpy as np
import faiss
import pickle
from pathlib import Path
from visionx.inference import FaceAnalysis

# --- Configuration ---
PROCESSED_DATA_DIR = "Processed_Data"
MODELS_DIR = "models"
YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "yolov8-face.onnx")
ARCFACE_MODEL_PATH = os.path.join(MODELS_DIR, "arcface_w600k_r50.onnx")

INDEX_FILE = "face_index.bin"
LABELS_FILE = "labels.pkl"
EMBEDDING_DIM = 512

def create_face_index():
    """
    Generates embeddings for all processed images and builds a FAISS index.
    """
    processed_path = Path(PROCESSED_DATA_DIR)
    if not processed_path.exists():
        print(f"❌ Error: Processed data directory not found at {PROCESSED_DATA_DIR}")
        return

    # Initialize the model analysis tool
    face_analyzer = FaceAnalysis(YOLO_MODEL_PATH, ARCFACE_MODEL_PATH)
    
    all_embeddings = []
    labels = []

    print("⏳ Generating embeddings from the processed dataset...")
    for person_folder in processed_path.iterdir():
        if not person_folder.is_dir():
            continue
        
        person_name = person_folder.name
        for image_file in person_folder.glob("*.jpg"):
            try:
                image = cv2.imread(str(image_file))
                # The images are already cropped, so we directly get the embedding
                embedding = face_analyzer.get_embedding(image)
                
                all_embeddings.append(embedding)
                labels.append(person_name)
            except Exception as e:
                print(f"Could not process {image_file}: {e}")

    if not all_embeddings:
        print("❌ No embeddings were generated. Exiting.")
        return

    # Convert to a NumPy array for FAISS
    all_embeddings_np = np.array(all_embeddings, dtype=np.float32)

    # Build the FAISS index
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(all_embeddings_np)
    
    # Save the index and the labels list
    faiss.write_index(index, INDEX_FILE)
    with open(LABELS_FILE, 'wb') as f:
        pickle.dump(labels, f)

    print(f"\n✅ Successfully created index for {index.ntotal} faces.")
    print(f"   -> Index saved to: {INDEX_FILE}")
    print(f"   -> Labels saved to: {LABELS_FILE}")

if __name__ == "__main__":
    create_face_index()