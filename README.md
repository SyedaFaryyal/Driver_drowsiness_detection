# Driver_drowsiness_detection
This project aims to detect signs of driver drowsiness in real-time using computer vision. The system leverages facial landmarks and eye aspect ratio (EAR) to monitor the driver’s eye movements and blinks to identify fatigue or drowsiness. If signs of drowsiness are detected, an alert is triggered to help prevent accidents.
🔍 Features:
Real-time video stream analysis using OpenCV

Eye aspect ratio (EAR) calculation for drowsiness detection

Audio alert when drowsiness is detected

Uses facial landmark detection with dlib or mediapipe

Lightweight and works on most modern computers

💡 Technologies Used:
Python

OpenCV

dlib or Mediapipe (for facial landmark detection)

NumPy

Pygame or playsound (for alert)

📌 How It Works:
The webcam captures the driver's face in real time.

Facial landmarks are detected to locate the eyes.

The Eye Aspect Ratio (EAR) is calculated — if the EAR falls below a threshold for several consecutive frames, it indicates that the driver may be drowsy.

A sound alert is played to wake the driver.
