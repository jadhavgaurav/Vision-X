# In src/visionx/inference.py

import onnxruntime as ort
import numpy as np
import cv2
from transformers import pipeline
from PIL import Image

class FaceAnalysis:
    def __init__(self, yolo_path: str, arcface_path: str, providers=None):
        if providers is None:
            providers = ['CUDAExecutionProvider','CPUExecutionProvider']
        
        # --- Load ONNX models from file ---
        self.face_detector = ort.InferenceSession(yolo_path, providers=providers)
        self.face_recognizer = ort.InferenceSession(arcface_path, providers=providers)

        # --- Models loaded via Transformers ---
        device = 0 if "CUDAExecutionProvider" in providers else -1
        
        print("Initializing Expression Recognition pipeline...")
        self.expression_pipeline = pipeline(
            "image-classification", 
            model="trpakov/vit-face-expression",
            device=device
        )
        print("Expression Recognition pipeline ready.")

        # Initialize the new age detection pipeline
        print("Initializing Age Detection pipeline...")
        self.age_pipeline = pipeline(
            "image-classification",
            model="nateraw/vit-age-classifier",
            device=device
        )
        print("Age Detection pipeline ready.")

        # --- Model Input Names ---
        self.detector_input_name = self.face_detector.get_inputs()[0].name
        self.recognizer_input_name = self.face_recognizer.get_inputs()[0].name

    def detect_faces(self, image: np.ndarray):
        img_height, img_width, _ = image.shape
        input_height, input_width = 640, 640
        scale = min(input_width / img_width, input_height / img_height)
        new_width, new_height = int(img_width * scale), int(img_height * scale)
        resized_img = cv2.resize(image, (new_width, new_height))
        padded_img = np.full((input_height, input_width, 3), 114, dtype=np.uint8)
        padded_img[(input_height - new_height) // 2:(input_height + new_height) // 2, (input_width - new_width) // 2:(input_width + new_width) // 2] = resized_img
        blob = cv2.dnn.blobFromImage(padded_img, 1/255.0, (input_width, input_height), swapRB=True)
        outputs = self.face_detector.run(None, {self.detector_input_name: blob})[0].T
        boxes = []
        for row in outputs:
            confidence = row[4]
            if confidence > 0.5:
                cx, cy, w, h = row[:4]
                x_offset = (input_width - new_width) / 2
                y_offset = (input_height - new_height) / 2
                x1, y1, x2, y2 = int((cx-w/2-x_offset)/scale), int((cy-h/2-y_offset)/scale), int((cx+w/2-x_offset)/scale), int((cy+h/2-y_offset)/scale)
                boxes.append([x1, y1, x2, y2])
        return boxes

    def get_embedding(self, face_image: np.ndarray) -> np.ndarray:
        face_image_resized = cv2.resize(face_image, (112, 112))
        blob = cv2.dnn.blobFromImage(face_image_resized, 1.0/127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        embedding = self.face_recognizer.run(None, {self.recognizer_input_name: blob})[0]
        norm_embedding = embedding / np.linalg.norm(embedding)
        return norm_embedding.flatten()

    def get_expression(self, face_image: np.ndarray) -> str:
        face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(face_image_rgb)
        predictions = self.expression_pipeline(pil_image)
        return predictions[0]['label']
    
    def get_age(self, face_image: np.ndarray) -> str:
        face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(face_image_rgb)
        predictions = self.age_pipeline(pil_image)
        return f"Age: {predictions[0]['label']}"