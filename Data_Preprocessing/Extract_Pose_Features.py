import os
import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
mp_drawing = mp.solutions.drawing_utils

# Input and output folders
dataset_path = r"D:\S11\S11_Phoning_2.55011271_002156.jpg"
output_file = r"pose_features.npy"

pose_features = []
labels = []

# Iterate through the dataset
for label in os.listdir(dataset_path):
    action_folder = os.path.join(dataset_path, label)
    image = cv2.imread(action_folder)
    if image is None:
        continue
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
        pose_features.append(landmarks.flatten())
        labels.append(label)

        annotated_image = image.copy()
        mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow("Pose Landmarks", annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

#Save features and labels
np.savez("pose_data.npz", features=np.array(pose_features), labels=np.array(labels))
