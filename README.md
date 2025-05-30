# Liveness Detection

A real-time computer vision system that detects human liveness using **facial landmarks**, **eye blink detection**, and **mouth expression velocity**. Built with Google’s MediaPipe Tasks API and visualized with matplotlib.

---

## 🧠 Features

- Tracks 478 facial landmarks using `face_landmarker.task`
- Detects **eye blinks** using vertical eyelid distance
- Measures **mouth opening and expression velocity**
- Displays a **live graph** of mouth and eye dynamics

---

## 📦 Methods & Tools

- **MediaPipe Tasks API** – high-precision facial landmark model
- **Matplotlib** – live visualization of tracked facial metrics
- **OpenCV** – webcam feed and face tracking
- **NumPy** – geometric calculations

---

## 📂 Project Structure

# Getting Started

### Windows

#### Open bash
Enter command below in bash terminal


##### Create a virtual environment
```bash
python -m venv .venv
```

##### Activate the environment
```bash
.venv\Scripts\activate
```

##### Install dependencies
```bash
pip install -r dependencies.txt
```


### macOS/Linux

##### Create a virtual environment
```bash
python3 -m venv .venv
```

##### Activate the environment
```bash
source .venv/bin/activate
```

---

### 🔄 Clone the repository

```bash
git clone https://github.com/wizdaksh/livenessDetection
cd livenessDetection
```
---

### Install dependencies
```bash
pip install -r dependencies.txt
```

---

### Import libraries from dependecies.txt (if not added)
```python
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from collections import deque
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import GestureRecognizer, GestureRecognizerOptions
import random
import csv
import time
```
