# In visionx/data_utils.py

import os
import cv2
import numpy as np
import albumentations as A
from pathlib import Path
from PIL import Image
import pillow_heif
import rawpy

from .inference import FaceAnalysis

IMG_SIZE = (112, 112)

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.ColorJitter(p=0.3),
    A.GaussNoise(p=0.2),
])

def read_image_universal(path: Path) -> np.ndarray:
    ext = path.suffix.lower()
    try:
        if ext in ['.heic', '.heif']:
            heif_file = pillow_heif.read_heif(path)
            image = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, "raw")
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        elif ext in ['.cr2']:
            with rawpy.imread(str(path)) as raw:
                return raw.postprocess()
        else:
            return cv2.imread(str(path))
    except Exception as e:
        print(f"Error reading {path.name} with custom loader: {e}")
        return None

def process_and_augment_dataset(raw_data_dir: str, processed_data_dir: str, face_analyzer: FaceAnalysis, augmentations_per_image: int = 5):
    raw_path = Path(raw_data_dir)
    processed_path = Path(processed_data_dir)
    print("Starting dataset processing...")

    for person_folder in raw_path.iterdir():
        if not person_folder.is_dir(): continue
        person_name = person_folder.name.lower()
        output_person_folder = processed_path / person_name
        output_person_folder.mkdir(parents=True, exist_ok=True)
        
        image_counter = 1
        for image_file in person_folder.iterdir():
            print(f"Processing {image_file.name}...")
            image = read_image_universal(image_file)
            if image is None:
                print(f"  -> Warning: Could not read image, skipping.")
                continue

            boxes = face_analyzer.detect_faces(image)
            if not boxes:
                print(f"  -> Warning: No face detected, skipping.")
                continue
            
            box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            x1, y1, x2, y2 = [max(0, int(val)) for val in box]
            
            cropped_face = image[y1:y2, x1:x2]
            if cropped_face.size == 0:
                print(f"  -> Warning: Invalid crop dimensions, skipping.")
                continue

            resized_face = cv2.resize(cropped_face, IMG_SIZE)
            
            base_filename = f"{person_name}-{image_counter:02d}.jpg"
            cv2.imwrite(str(output_person_folder / base_filename), resized_face)

            for i in range(augmentations_per_image):
                augmented = transform(image=resized_face)
                aug_filename = f"{person_name}-{image_counter:02d}-aug-{i+1}.jpg"
                cv2.imwrite(str(output_person_folder / aug_filename), augmented['image'])
            
            image_counter += 1

    print("\n✅ Dataset processing and augmentation complete.")
    print(f"Processed data saved in: {processed_data_dir}")