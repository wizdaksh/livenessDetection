import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import threading
from collections import deque
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

# Load MediaPipe model
model_path = r"C:\Users\daksh\Documents\Projects\Authentication\src\models\face_landmarker.task"
options = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    output_face_blendshapes=False,
    num_faces=1,
    running_mode=vision.RunningMode.VIDEO
)
face_landmarker = vision.FaceLandmarker.create_from_options(options)

# Shared data buffers
max_len = 100
mouth_openings = deque(maxlen=max_len)
mouth_velocities = deque(maxlen=max_len)
left_eye_dists = deque(maxlen=max_len)
right_eye_dists = deque(maxlen=max_len)

def live_plot():
    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    fig.tight_layout(pad=3)

    while True:
        ax1.clear()
        ax2.clear()
        ax1.set_title("Mouth Opening & Velocity")
        ax2.set_title("Eye Distance (Vertical)")

        ax1.plot(mouth_openings, label="Mouth Opening")
        ax1.plot(mouth_velocities, label="Velocity")
        ax1.legend(loc="upper right")

        ax2.plot(left_eye_dists, label="Left Eye")
        ax2.plot(right_eye_dists, label="Right Eye")
        ax2.legend(loc="upper right")

        plt.pause(0.05)

# Start plot thread
plot_thread = threading.Thread(target=live_plot, daemon=True)
plot_thread.start()

# Video processing
cap = cv2.VideoCapture(0)
frame_idx = 0
prev_mouth_opening = None

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.resize(frame, (640, 480))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = face_landmarker.detect_for_video(mp_image, frame_idx * 33)

    if result.face_landmarks:
        ih, iw, _ = frame.shape
        lm = result.face_landmarks[0]

        def to_xy(idx):
            return int(lm[idx].x * iw), int(lm[idx].y * ih)

        def dist(idx1, idx2):
            x1, y1 = to_xy(idx1)
            x2, y2 = to_xy(idx2)
            return np.linalg.norm([x1 - x2, y1 - y2])

        # Mouth
        top_y = lm[13].y * ih
        bottom_y = lm[14].y * ih
        opening = abs(bottom_y - top_y)
        mouth_openings.append(opening)

        velocity = abs(opening - prev_mouth_opening) if prev_mouth_opening else 0
        mouth_velocities.append(velocity)
        prev_mouth_opening = opening

        # Eyes
        left_eye = dist(386, 374)
        right_eye = dist(159, 145)
        left_eye_dists.append(left_eye)
        right_eye_dists.append(right_eye)

        # Draw
        for idx in [13, 14, 386, 374, 159, 145]:
            cv2.circle(frame, to_xy(idx), 2, (0, 255, 0), -1)

    cv2.imshow("Face Tracking", frame)
    frame_idx += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
