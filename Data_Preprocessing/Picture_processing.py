import os
import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

# Input and output folders
dataset_path = "path/to/dataset"
output_file = "pose_features.npy"

pose_features = []
labels = []

# Iterate through the dataset
for label in os.listdir(dataset_path):
    action_folder = os.path.join(dataset_path, label)
    if os.path.isdir(action_folder):
        for img_name in os.listdir(action_folder):
            img_path = os.path.join(action_folder, img_name)
            image = cv2.imread(img_path)
            if image is None:
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            if results.pose_landmarks:
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
                pose_features.append(landmarks.flatten())
                labels.append(label)

# Save features and labels
np.savez("pose_data.npz", features=np.array(pose_features), labels=np.array(labels))
