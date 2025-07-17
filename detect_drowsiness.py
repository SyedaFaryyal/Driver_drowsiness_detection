import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pyttsx3
import time
import threading  # Added for async audio

model = load_model("drowsiness_ann_model.h5")
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Initialize engine once globally
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Function to speak without blocking
def speak_async(text):
    def run():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run, daemon=True).start()

cap = cv2.VideoCapture(0)
print("[INFO] Starting webcam...")

blink_counter = 0
eye_closed_frames = 0
blink_threshold = 2
drowsy_start_time = None
drowsy_triggered = False

EAR_CLOSED_THRESHOLD = 0.2
EAR_DROWSY_SECONDS = 2
MAR_THRESHOLD = 0.6
MAR_DROWSY_THRESHOLD = 0.6
YAWN_DURATION = 1

yawn_start_time = None

def get_aspect_ratio(landmarks, points):
    A = np.linalg.norm(landmarks[points[1]] - landmarks[points[5]])
    B = np.linalg.norm(landmarks[points[2]] - landmarks[4])
    C = np.linalg.norm(landmarks[0] - landmarks[3])
    return (A + B) / (2.0 * C)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    brightness = np.mean(frame)
    if brightness < 40:
        cv2.putText(frame, "Low light detected!", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    label = "Unknown"
    color = (255, 255, 255)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            h, w, _ = frame.shape
            landmarks = np.array([[pt.x * w, pt.y * h] for pt in face_landmarks.landmark])

            left_eye_idx = [33, 160, 158, 133, 153, 144]
            right_eye_idx = [362, 385, 387, 263, 373, 380]
            mouth_idx = [78, 308, 13, 14, 312, 82]

            left_eye = landmarks[left_eye_idx]
            right_eye = landmarks[right_eye_idx]
            mouth = landmarks[mouth_idx]

            left_ear = get_aspect_ratio(left_eye, list(range(6)))
            right_ear = get_aspect_ratio(right_eye, list(range(6)))
            ear = (left_ear + right_ear) / 2.0
            mar = np.linalg.norm(mouth[2] - mouth[3]) / np.linalg.norm(mouth[0] - mouth[1])

            input_data = np.array([[ear, mar]])
            prediction = model.predict(input_data)[0][0]

            # EAR-based detection
            if ear < EAR_CLOSED_THRESHOLD:
                eye_closed_frames += 1
                if drowsy_start_time is None:
                    drowsy_start_time = time.time()
                elif not drowsy_triggered and time.time() - drowsy_start_time >= EAR_DROWSY_SECONDS:
                    speak_async("Wake up! You are drowsy.")  # Now non-blocking
                    drowsy_triggered = True
            else:
                if eye_closed_frames >= blink_threshold:
                    blink_counter += 1
                eye_closed_frames = 0
                drowsy_start_time = None
                drowsy_triggered = False

            # MAR-based detection
            if mar > MAR_DROWSY_THRESHOLD:
                if yawn_start_time is None:
                    yawn_start_time = time.time()
                elif time.time() - yawn_start_time >= YAWN_DURATION:
                    if not drowsy_triggered:
                        speak_async("Warning! You appear to be yawning excessively.")  # Non-blocking
                        drowsy_triggered = True
            else:
                yawn_start_time = None

            # Update status label
            if ear < EAR_CLOSED_THRESHOLD:
                label = "Drowsy"
                color = (0, 0, 255)
            elif mar > MAR_THRESHOLD:
                label = "Yawning"
                color = (0, 165, 255)
            else:
                label = "Alert"
                color = (0, 255, 0)

            # Display metrics
            cv2.putText(frame, f"EAR: {ear:.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Status: {label}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"Blinks: {blink_counter}", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)

    cv2.imshow("Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()