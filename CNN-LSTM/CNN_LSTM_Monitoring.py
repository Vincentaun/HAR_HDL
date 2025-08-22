import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque
import pickle

# --- 1. Load the Saved Model and Preprocessing Objects ---
# Load your final CNN-LSTM model
model = tf.keras.models.load_model("final_cnn_lstm_model2.h5")

# Load the scaler and label encoder (saved from your training process)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# --- 2. Define Parameters for Real-Time Monitoring ---
# Number of frames to form a sequence (should match training configuration)
sequence_length = 10
# Feature dimension: For MediaPipe Pose, by default 33 landmarks are detected.
# Each landmark has (x, y, z) coordinates, so feature_dim = 33 * 3 = 99.
feature_dim = 99

# Initialize a deque to store a sequence of frames
sequence_buffer = deque(maxlen=sequence_length)

# --- 3. Initialize MediaPipe Pose ---
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(static_image_mode=False,
                             min_detection_confidence=0.5,
                             min_tracking_confidence=0.5)

# --- 4. Define a Function to Extract and Preprocess Landmarks ---
def extract_landmarks(frame, pose_detector):
    """
    Process a frame to extract and flatten pose landmarks.
    Returns a 1D numpy array of landmarks if detected, otherwise None.
    """
    # Convert the frame to RGB (MediaPipe expects RGB images)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(rgb_frame)
    if results.pose_landmarks:
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            # Append x, y, z coordinates for each landmark
            landmarks.extend([lm.x, lm.y, lm.z])
        # Ensure we have the expected number of features (e.g., 99)
        if len(landmarks) == feature_dim:
            return np.array(landmarks)
    return None

# --- 5. Real-Time Monitoring Loop ---
cap = cv2.VideoCapture(1)  # Use your webcam (0); adjust if needed

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Extract landmarks from the current frame
    landmarks = extract_landmarks(frame, pose_detector)
    if landmarks is not None:
        # Preprocess landmarks using the saved scaler
        # Reshape to (1, feature_dim) for scaler transformation
        landmarks_scaled = scaler.transform([landmarks])[0]
        # Add the scaled landmarks to the sequence buffer
        sequence_buffer.append(landmarks_scaled)

        # If the buffer is full, perform prediction
        if len(sequence_buffer) == sequence_length:
            # Convert the sequence to a numpy array and reshape to model input:
            # Expected shape: (1, sequence_length, feature_dim, 1)
            input_sequence = np.array(sequence_buffer)
            input_sequence = input_sequence.reshape(1, sequence_length, feature_dim, 1)
            
            # Run the model prediction
            prediction_probs = model.predict(input_sequence)
            predicted_class = np.argmax(prediction_probs, axis=1)
            predicted_label = label_encoder.inverse_transform(predicted_class)[0]
            
            # Overlay the predicted action on the frame
            cv2.putText(frame, f"Action: {predicted_label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        # Clear the buffer if no landmarks are detected
        sequence_buffer.clear()
    
    # Display the real-time video with predictions
    cv2.imshow("Real-Time Action Recognition", frame)
    
    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
