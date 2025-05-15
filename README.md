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
python -m venv myenv
```

##### Activate the environment
```bash
myenv\Scripts\activate
```

##### Install dependencies
```bash
pip install -r requirements.txt
```


### macOS/Linux

##### Create a virtual environment
```bash
python3 -m venv myenv
```

##### Activate the environment
```bash
source myenv/bin/activate
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
pip install -r requirements.txt
```

---

### Import libraries from dependecies.txt (if not added)
```python
import mediapipe as mp 
import cv2
import matplotlib.pyplot as plt
import numpy as np
```
