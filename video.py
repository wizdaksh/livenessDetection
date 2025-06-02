#Initialize the libraries 
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import GestureRecognizer, GestureRecognizerOptions
import random



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
# face_model_path = r"livenessDetection\models\face_landmarker.task"
# gesture_model_path = r"livenessDetection\models\gesture_recognizer.task"

# --- MacOS/Linux path ---
face_model_path = "livenessDetection/models/face_landmarker.task"
gesture_model_path = "livenessDetection/models/gesture_recognizer.task"

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

# --- Define the codec and create VideoWriter object ---
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))


if not cap.isOpened():
    print("[ERROR] Failed to open webcam. Exiting.")
    exit()


# --- Default values ---
frame_idx = 0                             # Integer: current frame index
prev_mouth_opening = None                 # Float or None: previous mouth opening value
gesture_result_text = "None"              # String: name of the detected gesture

# --- Initialize valid gesture ---
valid_gesture = random.choice(gestureStringList)  # String: randomly chosen valid gesture from list




# loop runs if capturing has been initialized. 
while cap.isOpened():
    # frame frames from a camera 
    # success checks return at each frame
    success, frame = cap.read()           # Read a frame from webcam; 'frame' is a NumPy array (image)
    if not success:
        print("[ERROR] Failed to grab frame. Exiting.")
        break
    
    # Convert OpenCV BGR image to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)


    # output the frame
    out.write(rgb_frame) 
    


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


    frame_idx += 1
    
    # The original input frame is shown in the window 
    cv2.imshow('Original', frame)

    # The window showing the operated video stream 
    # cv2.imshow('frame', rgb_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




# Close the window / Release webcam
cap.release()

# After we release our webcam, we also release the output
out.release() 

# De-allocate any associated memory usage 
cv2.destroyAllWindows()
