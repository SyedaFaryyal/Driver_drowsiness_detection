
import numpy as np

def get_ear(eye_points):
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

def get_mar(mouth_points):
    A = np.linalg.norm(mouth_points[13] - mouth_points[19])
    B = np.linalg.norm(mouth_points[14] - mouth_points[18])
    C = np.linalg.norm(mouth_points[15] - mouth_points[17])
    D = np.linalg.norm(mouth_points[12] - mouth_points[16])
    mar = (A + B + C) / (2.0 * D)
    return mar
