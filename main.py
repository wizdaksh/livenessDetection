#Initialize the libraries 
import cv2
import numpy as np
import mediapipe as mp
# import matplotlib.pyplot as plt
from collections import deque
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import GestureRecognizer, GestureRecognizerOptions
import random
# import csv
# import time

# --- List of valid gestures ---
gestureStringList = [
    "ILoveYou",
    "Victory",
    "Thumb_Up",
    "Thumb_Down",
    "Pointing_Up",
    "Open_Palm",
    "Closed_Fist",
]

# # --- Model paths (Windows path) ---
face_model_path = r"livenessDetection\models\face_landmarker.task"
gesture_model_path = r"livenessDetection\models\gesture_recognizer.task"

# --- MacOS/Linux path ---
# face_model_path = "models/face_landmarker.task"
# gesture_model_path = "models/gesture_recognizer.task"

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

# # --- Initialize graph object ---
# plt.ion()
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
# fig.tight_layout(pad=3)

# --- Plot face data ---
# def plotFaceDeltas():
    # ax1.clear()
    # ax2.clear()
    # ax1.set_title("Mouth Opening & Velocity")
    # ax2.set_title("Eye Distance (Vertical)")
    # ax1.plot(mouth_openings, label="Mouth Opening")
    # ax1.plot(mouth_velocities, label="Velocity")
    # ax1.legend(loc="upper right")
    # ax2.plot(left_eye_dists, label="Left Eye")
    # ax2.plot(right_eye_dists, label="Right Eye")
    # ax2.legend(loc="upper right")
    # plt.pause(0.001)

# --- Buffers for live plots ---
max_len = 100
mouth_openings = deque(maxlen=max_len)  # deque (double-ended queue) to store recent mouth velocity values (floats)
mouth_velocities = deque(maxlen=max_len)  # deque (double-ended queue) to store recent mouth velocity values (floats)
left_eye_dists = deque(maxlen=max_len)    # deque to store recent left eye vertical distances (floats)
right_eye_dists = deque(maxlen=max_len)   # deque to store recent right eye vertical distances (floats)

# --- Default values ---
frame_idx = 0                             # Integer: current frame index
prev_mouth_opening = None                 # Float or None: previous mouth opening value
gesture_result_text = "None"              # String: name of the detected gesture

# --- Initialize valid gesture ---
valid_gesture = random.choice(gestureStringList)  # String: randomly chosen valid gesture from list

# start_time = time.perf_counter()  # Start time for performance measurement


# --- Lists for eylid and mouth pixel distance. Last in first out ---
human = False
eyeScore = 0
mouthScore = 0
attempts = 0

face_saved = False
# with open ("face_data.csv", mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['Time', 'Opening', 'Velocity', 'Left Eye', 'Right Eye'])
while True:
    success, frame = cap.read()           # Read a frame from webcam; 'frame' is a NumPy array (image)
    if not success:
        print("[ERROR] Failed to grab frame. Exiting.")
        break
    
    # Convert OpenCV BGR image to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # --- Face landmarks ---
    face_result = face_landmarker.detect_for_video(mp_image, frame_idx * 33)
    if face_result.face_landmarks:
        ih, iw, _ = frame.shape           # ih: image height, iw: image width
        lm = face_result.face_landmarks[0]  # List of landmark objects (each has .x and .y attributes, normalized [0,1])

        # Helper: convert normalized landmark index to pixel coordinates
        def to_xy(idx): 
            return int(lm[idx].x * iw), int(lm[idx].y * ih)

        # Helper: Euclidean distance between two landmarks (in pixels)
        def dist(idx1, idx2):
            x1, y1 = to_xy(idx1)
            x2, y2 = to_xy(idx2)
            # np.linalg.norm computes sqrt((x1-x2)^2 + (y1-y2)^2)
            return np.linalg.norm([x1 - x2, y1 - y2])

        # Calculate vertical mouth opening (in pixels)
        top_y = lm[13].y * ih             # y-coordinate of upper lip landmark
        bottom_y = lm[14].y * ih          # y-coordinate of lower lip landmark
        opening = abs(bottom_y - top_y)   # Absolute vertical distance between lips
        mouth_openings.append(opening)    # Store in deque

        # Calculate mouth opening velocity (change per frame)
        velocity = abs(opening - prev_mouth_opening) if prev_mouth_opening else 0
        mouth_velocities.append(velocity) # Store in deque
        prev_mouth_opening = opening      # Update for next frame

        # Calculate vertical distances for left and right eyes (in pixels)
        left_eye = dist(386, 374)         # Distance between two left eye landmarks
        right_eye = dist(159, 145)        # Distance between two right eye landmarks
        left_eye_dists.append(left_eye)   # Store in deque
        right_eye_dists.append(right_eye) # Store in deque


        # Draw circles on selected landmarks for visualization
        for idx in [13, 14, 386, 374, 159, 145]:
            cv2.circle(frame, to_xy(idx), 2, (0, 255, 0), -1)

        # elapsed_time = time.perf_counter() - start_time  # Calculate elapsed time
        # Write data to CSV file
        # writer.writerow([f"{elapsed_time:.4f}", f"{opening:.2f}", f"{velocity:.2f}", f"{left_eye:.2f}", f"{right_eye:.2f}"])

    # --- Gesture recognition ---
    gesture_result = gesture_recognizer.recognize_for_video(mp_image, frame_idx * 33)
    if gesture_result.gestures:
        gesture_result_text = gesture_result.gestures[0][0].category_name  # String: name of detected gesture
    else:
        gesture_result_text = "None"

    # Check if detected gesture matches the valid gesture
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

    mouth_openingsList = list(mouth_openings)  # Convert deque to list for plotting
    mouth_velocitiesList = list(mouth_velocities)  # Convert deque to list for plotting     
    left_eye_distsList = list(left_eye_dists)  # Convert deque to list for plotting
    right_eye_distsList = list(right_eye_dists)  # Convert deque to list for plotting
    


    #Check Eyelid Delta for Validity 
    if validity:
        #TODO: Check the ratio of max and min distances for eylids, should be less than 2.13
        if len(mouth_openingsList) > 20:
            lastTenMouthOpenings = mouth_openingsList[-10:]  # Get last 10 mouth openings
            lastTenMouthVelocities = mouth_velocitiesList[-10:]  # Get last 10 mouth velocities

            maxMouthOpening = max(lastTenMouthOpenings)
            minMouthOpening = min(lastTenMouthOpenings)
            mouthRatio = maxMouthOpening/ minMouthOpening
                    

            lastTenLeftEye = left_eye_distsList[-10:]  # Get last 10 left eye distances
            lastTenRightEye = right_eye_distsList[-10:]  # Get last 10 right eye distances

            maxLeftEyeDistance = max(lastTenLeftEye)
            maxRightEyeDistance = max(lastTenRightEye)                
            maxEyeAverage = (maxLeftEyeDistance + maxRightEyeDistance) / 2
    
            minLeftEyeDistance = min(lastTenLeftEye)
            minRightEyeDistance = min(lastTenRightEye)
            minEyeAverage = (minLeftEyeDistance + minRightEyeDistance) / 2
            
            eyeAverage = maxEyeAverage, minEyeAverage
            eyeRatio = maxEyeAverage / minEyeAverage

            cv2.putText(frame, f"Eye Ratio: {eyeRatio:.2f}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            if eyeRatio > 2.7:
                cv2.putText(frame, "Eyelid Delta: Valid", (10, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)         
                eyeScore += 1
        
            cv2.putText(frame, f"Mouth Ratio: {mouthRatio:.2f}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            if mouthRatio > 10:
                cv2.putText(frame, "Mouth Delta: Valid", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                mouthScore += 1
            
            if eyeScore > 1 and mouthScore > 0:
                human = True

            if human and not face_saved:
                cv2.putText(frame, "Human Detected", (10, 190),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # --- Crop face using landmarks ---
                if face_result.face_landmarks:
                    ih, iw, _ = frame.shape
                    lm = face_result.face_landmarks[0]

                    x_coords = [int(point.x * iw) for point in lm]
                    y_coords = [int(point.y * ih) for point in lm]

                    padding = 20
                    x_min = max(min(x_coords) - padding, 0)
                    x_max = min(max(x_coords) + padding, iw)
                    y_min = max(min(y_coords) - padding, 0)
                    y_max = min(max(y_coords) + padding, ih)

                    cropped_face = frame[y_min:y_max, x_min:x_max]
                    cv2.imwrite("cropped_face.jpg", cropped_face)
                    cv2.imshow("Cropped Face", cropped_face)

                    face_saved = True  # Prevent further saving

                            
                else: 
                    human = False
                    eyeScore = 0    
                    mouthScore = 0
                    face_saved = False  # Reset so you can save again next time human is detected

            
    # Plot Live Data of Eyelid Delta and Lip Delta Velocity
    # plotFaceDeltas()  
    
    # Show frame
    cv2.imshow("Face + Gesture Tracking", frame)
    frame_idx += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()
# plt.ioff()
# plt.close()
