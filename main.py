import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from collections import deque
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import GestureRecognizer, GestureRecognizerOptions
import random

gestureStringList = [
    "ILoveYou",
    "Victory",
    "Tumb_Up",
    "Thumb_Down",
    "Pointing_Up",
    "Open_Palm",
    "Closed_Fist",
    "None"
]

# --- Model paths ---
face_model_path = r"C:\Users\daksh\Documents\Projects\Authentication\src\models\face_landmarker.task"
gesture_model_path = r"C:\Users\daksh\Documents\Projects\Authentication\src\models\gesture_recognizer.task"

# --- Load face landmark model ---
face_options = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=face_model_path),
    output_face_blendshapes=False,
    num_faces=1,
    running_mode=vision.RunningMode.VIDEO
)
face_landmarker = vision.FaceLandmarker.create_from_options(face_options)

# --- Load gesture recognizer model ---
gesture_options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=gesture_model_path),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1
)
gesture_recognizer = GestureRecognizer.create_from_options(gesture_options)

# --- Start webcam (DirectShow fixes MSMF error) ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("[ERROR] Failed to open webcam. Exiting.")
    exit()

#Initialize graph object
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
fig.tight_layout(pad=3)

def plotFaceDeltas():
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
    plt.pause(0.001)

# --- Buffers for live plots ---
max_len = 100
mouth_openings = deque(maxlen=max_len)
mouth_velocities = deque(maxlen=max_len)
left_eye_dists = deque(maxlen=max_len)
right_eye_dists = deque(maxlen=max_len)

frame_idx = 0
prev_mouth_opening = None
gesture_result_text = "None"
valid_gesture = random.choice(gestureStringList)


while True:
    success, frame = cap.read()
    if not success:
        print("[ERROR] Failed to grab frame. Exiting.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # --- Face landmarks ---
    face_result = face_landmarker.detect_for_video(mp_image, frame_idx * 33)
    if face_result.face_landmarks:
        ih, iw, _ = frame.shape
        lm = face_result.face_landmarks[0]

        def to_xy(idx): return int(lm[idx].x * iw), int(lm[idx].y * ih)
        def dist(idx1, idx2):
            x1, y1 = to_xy(idx1)
            x2, y2 = to_xy(idx2)
            return np.linalg.norm([x1 - x2, y1 - y2])

        top_y = lm[13].y * ih
        bottom_y = lm[14].y * ih
        opening = abs(bottom_y - top_y)
        mouth_openings.append(opening)
        velocity = abs(opening - prev_mouth_opening) if prev_mouth_opening else 0
        mouth_velocities.append(velocity)
        prev_mouth_opening = opening

        left_eye = dist(386, 374)
        right_eye = dist(159, 145)
        left_eye_dists.append(left_eye)
        right_eye_dists.append(right_eye)

        for idx in [13, 14, 386, 374, 159, 145]:
            cv2.circle(frame, to_xy(idx), 2, (0, 255, 0), -1)

    # --- Gesture recognition ---
    gesture_result = gesture_recognizer.recognize_for_video(mp_image, frame_idx * 33)
    if gesture_result.gestures:
        gesture_result_text = gesture_result.gestures[0][0].category_name  
    else:
        gesture_result_text = "None"


    if gesture_result_text == valid_gesture:
            validity = True
            validity_text = "Valid Gesture"
    else:
        validity = False
        validity_text = "Invalid Gesture"

    
    # Draw gesture result
    cv2.putText(frame, f'Current Gesture: {gesture_result_text}', (10, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, f'Valid Gesture: {valid_gesture}', (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, validity_text, (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if validity else (0, 0, 255), 2)
  
    # Plot Live Data of Eyelid Delta and Lip Delta Velocity
    plotFaceDeltas()

    # Show frame
    cv2.imshow("Face + Gesture Tracking", frame)
    frame_idx += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.close()
