import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial import distance as dist
from imutils import face_utils
from collections import deque
import os

# Load Dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
model_path = os.path.join(os.path.dirname(__file__), 'models', 'shape_predictor_68_face_landmarks.dat')
predictor = dlib.shape_predictor(model_path)

# Get landmark indexes for eyes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# EAR Thresholds and Counters
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 2
blink_count = 0
frame_counter = 0

# Store last 100 EAR values for the running graph
ear_values = deque(maxlen=100)

# Function to compute Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Start Video Capture (Webcam)
camera = cv2.VideoCapture(0)  # 0 for default camera

# Setup Matplotlib Figure for EAR Graph
fig, ax = plt.subplots()
x_data = np.arange(0, 100)  # X-axis (last 100 frames)
y_data = np.zeros(100)      # Y-axis (EAR values)
line, = ax.plot(x_data, y_data, 'g-', label="EAR")

# Set Graph Labels
ax.set_ylim(0, 0.5)
ax.set_xlim(0, 100)
ax.set_xlabel("Frames")
ax.set_ylabel("EAR Value")
ax.set_title("Eye Aspect Ratio (EAR) Over Time")
ax.legend()

# Function to update the graph in real time
def update_graph(frame):
    global blink_count, frame_counter

    ret, img = camera.read()
    if not ret:
        return line,

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    ear = 0  # Default EAR value
    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        left_eye = shape[lStart:lEnd]
        right_eye = shape[rStart:rEnd]
        face = shape

        left_EAR = eye_aspect_ratio(left_eye)
        right_EAR = eye_aspect_ratio(right_eye)
        ear = (left_EAR + right_EAR) / 2.0  # Average EAR

        # Blink detection
        if ear < EAR_THRESHOLD:
            frame_counter += 1
        else:
            if frame_counter >= EAR_CONSEC_FRAMES:
                blink_count += 1
            frame_counter = 0  # Reset counter

        # Draw eye landmarks
        cv2.drawContours(img, [cv2.convexHull(left_eye)], -1, (0, 255, 0), 1)
        cv2.drawContours(img, [cv2.convexHull(right_eye)], -1, (0, 255, 0), 1)

        for (x, y ) in shape:
            cv2.circle(img, (x, y), 1, (0, 0, 255), -1)

    # Update EAR values for plotting
    ear_values.append(ear)
    line.set_ydata(np.pad(ear_values, (100 - len(ear_values), 0), 'constant'))
    
    # Show Blink Count on Screen
    cv2.putText(img, f"Blinks: {blink_count}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Show Webcam Feed
    cv2.imshow("Eye Blink Detection", img)


    # Stop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        camera.release()
        cv2.destroyAllWindows()
        plt.close()

    return line,

# Use Matplotlib's Animation to Update Graph
ani = animation.FuncAnimation(fig, update_graph, interval=50, blit=False)

# Show Graph in Separate Window
plt.show()

# Cleanup
camera.release()
cv2.destroyAllWindows()
