import joblib
import mediapipe as mp
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load the CSV file
data = pd.read_csv(r"C:\Users\User\Desktop\python\113-1 AI Practice\H36M\pose_coordinates.csv")

# Separate features and labels
X = data.iloc[:, :-1].values  # All columns except the last
y = data.iloc[:, -1].values   # Last column is the action label

# Encode labels (if necessary)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Load the trained Random Forest model
rf_model = joblib.load("random_forest_model.pkl")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

# Open webcam or video file
cap = cv2.VideoCapture(1)  # Use 0 for webcam, or provide a video file path

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to RGB for Mediapipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    
    if results.pose_landmarks:
        # Extract pose coordinates
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
        features = landmarks.flatten()  # Flatten to 1D array

        numerical_label = rf_model.predict(features.reshape(1,-1))[0]
        # Access the mapping
        class_mapping = dict(zip(range(len(label_encoder.classes_)), label_encoder.classes_))
        # Convert numerical label to action name
        action_label = class_mapping[numerical_label]

        # Overlay action label on the frame
        cv2.putText(frame, f"Action: {action_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Real-Time Action Recognition", frame)
    
    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
