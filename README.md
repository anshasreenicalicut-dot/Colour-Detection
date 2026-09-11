# 🚗 Car Colour Detection & Traffic Counter

## 📌 Project Overview

Car Colour Detection & Traffic Counter is a Python-based computer vision project that uses **YOLOv8, OpenCV, NumPy, and Tkinter** to analyze traffic images.

The system can:

- 🚘 Detect cars in traffic
- 🎨 Detect the colour of cars
- 🔴 Draw a **red bounding box for blue cars**
- 🔵 Draw a **blue bounding box for other coloured cars**
- 👤 Detect and count people
- 🚦 Detect traffic signals/lights
- 📊 Display the total number of cars and people
- 🖼️ Provide a GUI for uploading and previewing images

---

## ✨ Features

### 🚘 Car Detection
YOLOv8 is used to detect cars from traffic images.

### 🎨 Car Colour Detection
The project uses **HSV-based image processing** to identify the dominant colour of each detected car.

Supported colours include:

- Blue
- Red
- Green
- Yellow
- Orange
- Purple
- White
- Black
- Gray
- Silver
- Other

### 🟥 Bounding Box Rule

The project follows the required colour-box rule:

| Car Colour | Bounding Box |
|------------|--------------|
| 🔵 Blue | 🔴 Red |
| Other colours | 🔵 Blue |

### 👤 Person Detection

The system detects people in the traffic image and displays the total number of people detected.

### 🚦 Traffic Signal Detection

YOLOv8 can also detect traffic lights/signals in the input image.

### 🖥️ GUI

The application provides a graphical interface where users can:

1. Upload an image
2. Preview the input image
3. Run detection
4. View the processed image
5. See the number of cars
6. See the number of people

---

## 🛠️ Technologies Used

- Python 3.9+
- YOLOv8
- Ultralytics
- OpenCV
- NumPy
- Tkinter
- Pillow
- PyTorch

---

## 📂 Project Structure

```text
Car Colour Detection/
│
├── app.py
├── detect.py
├── color_detector.py
├── train.py
├── test_detection.py
├── data.yaml
├── test.jpg
├── yolov8n.pt
│
├── models/
│   └── best.pt
│
├── bdd100k/
│   ├── images/
│   │   └── 100k/
│   │       ├── train/
│   │       ├── val/
│   │       └── test/
│   │
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
│
└── runs/
    └── car_detection/
