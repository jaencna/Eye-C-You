import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import serial
import tkinter as tk
from PIL import Image, ImageTk

# Constants
EAR_THRESHOLD = 0.25
CONSECUTIVE_FRAMES = 3
WARNING_TIME = 50  # Frames for yellow LED
ALERT_TIME = 100  # Frames for red LED

# Set up serial communication with Arduino
arduino = serial.Serial('COM5', 9600)

# Function to calculate Eye Aspect Ratio (EAR)
def calculate_ear(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Function to calculate head pose
def calculate_head_pose(landmarks, frame_shape):
    size = frame_shape[:2]
    focal_length = size[1]
    center = (size[1] // 2, size[0] // 2)
    camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")
    dist_coeffs = np.zeros((4, 1))  # No distortion

    # 3D model points of a face
    model_points = np.array([
        (0.0, 0.0, 0.0),        # Nose tip
        (0.0, -330.0, -65.0),   # Chin
        (-225.0, 170.0, -135.0), # Left eye left corner
        (225.0, 170.0, -135.0), # Right eye right corner
        (-150.0, -150.0, -125.0), # Left Mouth corner
        (150.0, -150.0, -125.0)  # Right Mouth corner
    ])

    image_points = np.array([
        (landmarks.part(30).x, landmarks.part(30).y),  # Nose tip
        (landmarks.part(8).x, landmarks.part(8).y),   # Chin
        (landmarks.part(36).x, landmarks.part(36).y), # Left eye left corner
        (landmarks.part(45).x, landmarks.part(45).y), # Right eye right corner
        (landmarks.part(48).x, landmarks.part(48).y), # Left Mouth corner
        (landmarks.part(54).x, landmarks.part(54).y)  # Right Mouth corner
    ], dtype="double")

    # SolvePnP to estimate pose
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs
    )
    return rotation_vector, translation_vector

# Function to preprocess frame for low-light enhancement
def preprocess_frame(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Increase brightness and contrast
    alpha = 1.5  # Contrast control (1.0-3.0)
    beta = 30    # Brightness control (0-100)
    enhanced_gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

    # Apply histogram equalization
    equalized_gray = cv2.equalizeHist(enhanced_gray)

    return equalized_gray

# Load Dlib models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Eye landmarks
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))

# Video capture
cap = cv2.VideoCapture(2)

# Variables
frame_counter = 0
eye_closed_duration = 0
eye_closed = False
streaming = False  # Flag to control the stream

# GUI setup
root = tk.Tk()
root.title("Eye Closure Detection")

# Canvas for displaying video
canvas = tk.Canvas(root, width=640, height=480, bg="black")
canvas.pack()

status_label = tk.Label(root, text="Status: Click 'Start Stream' to begin", font=("Helvetica", 16))
status_label.pack()

def start_stream():
    global streaming
    streaming = True
    status_label.config(text="Status: Streaming started...", fg="blue")
    update_frame()

def stop_stream():
    global streaming
    streaming = False
    status_label.config(text="Status: Stream stopped", fg="black")
    canvas.delete("all")  # Clear the canvas

def update_frame():
    global frame_counter, eye_closed_duration, eye_closed

    if not streaming:
        return

    ret, frame = cap.read()
    if not ret:
        status_label.config(text="Error: Unable to capture video frame", fg="red")
        return

    # Preprocess the frame for low-light enhancement
    gray = preprocess_frame(frame)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # Head pose estimation
        rotation_vector, translation_vector = calculate_head_pose(landmarks, frame.shape)

        left_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in LEFT_EYE])
        right_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in RIGHT_EYE])

        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        if avg_ear < EAR_THRESHOLD:
            frame_counter += 1
            if frame_counter >= CONSECUTIVE_FRAMES:
                eye_closed = True
                eye_closed_duration += 1
        else:
            frame_counter = 0
            eye_closed = False
            eye_closed_duration = 0

        if eye_closed:
            if eye_closed_duration >= ALERT_TIME:
                arduino.write(b'2')  # Red LED and continuous buzzer
                status_label.config(text="Status: Eyes Closed - ALERT (Red LED)", fg="red")
            elif eye_closed_duration >= WARNING_TIME:
                arduino.write(b'1')  # Yellow LED and warning buzzer
                status_label.config(text="Status: Eyes Closed - WARNING (Yellow LED)", fg="orange")
            else:
                arduino.write(b'0')  # No alert
                status_label.config(text="Status: Eyes Closed", fg="blue")
        else:
            arduino.write(b'0')  # Reset
            status_label.config(text="Status: Eyes Open", fg="green")

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = ImageTk.PhotoImage(Image.fromarray(img))
    canvas.create_image(0, 0, anchor=tk.NW, image=img)
    canvas.imgtk = img

    root.after(10, update_frame)

# Buttons
button_frame = tk.Frame(root)
button_frame.pack()

start_button = tk.Button(button_frame, text="Start Stream", command=start_stream, bg="green", fg="white", font=("Helvetica", 12))
start_button.pack(side=tk.LEFT, padx=10)

stop_button = tk.Button(button_frame, text="Stop Stream", command=stop_stream, bg="red", fg="white", font=("Helvetica", 12))
stop_button.pack(side=tk.LEFT, padx=10)

# Start GUI loop
root.mainloop()

# Release resources
cap.release()
