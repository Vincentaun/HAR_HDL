import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from keras._tf_keras.keras.models import load_model
from sklearn.preprocessing import LabelEncoder


# Load the CSV file
data = pd.read_csv(r"C:\Users\User\Desktop\python\113-1 AI Practice\H36M\pose_coordinates.csv")

# Separate features and labels
X = data.iloc[:, :-1].values  # All columns except the last
y = data.iloc[:, -1].values   # Last column is the action label

# Encode labels (if necessary)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


# Load the trained CNN model
model = load_model("pose_action_model.h5")

# Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Start video capture
cap = cv2.VideoCapture(1)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame for pose detection
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        # Extract pose landmarks
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
        features = landmarks.flatten().reshape(1, -1, 1)  # Reshape for model

        # Predict action
        prediction = model.predict(features)
        class_mapping = dict(zip(range(len(label_encoder.classes_)), label_encoder.classes_))
        action_label = np.argmax(prediction)
        action = class_mapping[action_label] 

        # Display action
        cv2.putText(frame, f"Action: {action}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Action Detection', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
