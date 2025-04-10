# Liveness Detection

A computer vision system to detect real human presence and prevent spoofing attacks using photos, videos, and deepfakes.

---

## Overview

This project implements **liveness detection** techniques to distinguish between real faces and spoofed ones, such as:

- Printed photographs
- Video replays
- Deepfake manipulations

It's designed to enhance security in face authentication systems.

---

## Methods Used

- **Optical Flow** – Tracks motion to detect inconsistencies in fake media.
- **DLib Blink Detection** – Identifies natural blinking patterns as a sign of life.
- **68 Face Landmarks** - Returns x,y coordinates for facial landmarks
---

## Getting Started

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

### Import libraries (if not added)
```python
import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial import distance as dist
from imutils import face_utils
from collections import deque
import mediapipe as mp 
from mediapipe.tasks import python 
from mediapipe.tasks.python import vision
```
