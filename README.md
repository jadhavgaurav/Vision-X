# VisionX: Real-Time Facial Attendance & Analysis System

VisionX is a complete, real-time face recognition and analysis system built in Python. It serves as an intelligent attendance and security solution, capable of identifying individuals, tracking their sessions, analyzing demographic attributes, and issuing alerts for specific events. The application is built with a high-performance, multi-threaded architecture and features a GPU-accelerated desktop dashboard.

![VisionX Dashboard](https://i.imgur.com/qU3G7zL.png) 
*(Feel free to replace this with a screenshot of your own running application)*

## Features

-   **Real-Time Face Recognition**: Identifies multiple known individuals from a live video stream.
-   **Advanced Facial Analysis**: Performs real-time **expression recognition** and **age estimation**.
-   **Session Tracking & Logging**: Automatically logs entry/exit times and the direction of exit for each person to an **SQLite database**.
-   **Person Re-identification**: A smart cache maintains a person's identity if they temporarily leave and re-enter the frame, preventing duplicate logs.
-   **Alerting System**: Provides on-screen alerts for unknown individuals and console alerts for unauthorized exits.
-   **High-Performance GUI**: A multi-threaded desktop dashboard built with **Dear PyGui** ensures a smooth, non-blocking user experience with GPU-accelerated rendering.

## Technology Stack

-   **Core Language**: Python
-   **GUI Framework**: Dear PyGui
-   **AI & Computer Vision**:
    -   **Face Detection**: YOLOv8 (via ONNX)
    -   **Face Recognition**: ArcFace (via ONNX)
    -   **Expression & Age Analysis**: Hugging Face Transformers (PyTorch backend)
    -   **High-Speed Search**: FAISS (Facebook AI Similarity Search)
-   **Data & Processing**: OpenCV, Pandas, NumPy, Albumentations
-   **Database**: SQLite
-   **Dependency Management**: Poetry

## Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-link>
    cd VisionX
    ```

2.  **Install Dependencies:** This project uses Poetry for dependency management. Ensure you have Poetry installed.
    ```bash
    poetry install
    ```
    This command will create a virtual environment and install all necessary packages from the `pyproject.toml` and `poetry.lock` files.

## Usage

The project requires a one-time data setup before running the main application.

### Step 1: Add Face Images

-   Place images of known individuals into the `data/raw/` directory. Each person must have their own subfolder named after them.
-   For example: `data/raw/Gaurav/image1.jpg`, `data/raw/Priya/photo.png`.
-   For best results, use 5-10 high-quality images per person with varied angles and lighting.

### Step 2: Process Data and Build Face Index

-   Run the data preparation script to process and augment the images:
    ```bash
    poetry run python scripts/prepare_data.py
    ```
-   Run the indexing script to create the face database:
    ```bash
    poetry run python scripts/build_index.py
    ```
    This will generate `face_index.bin` and `labels.pkl` in the `resources/` folder.

### Step 3: Run the Main Application

-   Launch the Dear PyGui desktop dashboard:
    ```bash
    poetry run python app_dearpygui.py
    ```

## Project Architecture

The application uses a multi-threaded producer-consumer architecture to maintain a high-performance, non-blocking GUI.

1.  **Main Thread**: Handles the Dear PyGui interface, video capture from OpenCV, and rendering the UI. It captures a frame and places it in a queue.
2.  **Worker Thread**: A dedicated background thread continuously pulls frames from the queue. It performs all heavy AI inference tasks (detection, recognition, expression, age) and session logic.
3.  **Data Flow**: The worker thread sends the processed results (bounding boxes, names, analysis) back to the main thread via another queue, which then draws the information on the video feed. This ensures that the AI processing never freezes the user interface.