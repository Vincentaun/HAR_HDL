import os
import cv2
import mediapipe as mp
import numpy as np
import csv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

# Paths
source_folders = [r"D:\S1", r"D:\S5", r"D:\S6", r"D:\S7", r"D:\S8", r"D:\S9", r"D:\S11"]  # Replace with your folder paths
output_csv = r"C:\Users\User\Desktop\python\113-1 AI Practice\H36M\All_pose_landmarks.csv"

for folder in source_folders:

    # Path to dataset folder and output CSV file
    dataset_path = folder
    # Open CSV file to write the coordinates and labels
    with open(output_csv, mode="w", newline="", buffering=1) as file:
        writer = csv.writer(file)
        
        # Write header
        header = ["x" + str(i) for i in range(33)] + ["y" + str(i) for i in range(33)] + ["z" + str(i) for i in range(33)] + ["Action"]
        writer.writerow(header)
        # Iterate through images in the folder
        for img_name in os.listdir(dataset_path):
            img_path = os.path.join(dataset_path, img_name)
            # Process only images
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                print(f"Skipping non-image file: {img_name}")
                continue

            # Extract the action from the filename (e.g., S11_Eating.60457274_002126.jpg -> Eating)
            action = img_name.split('_')[1].split('.')[0] if '_' in img_name else "unknown"
            # Read and process the image
            image = cv2.imread(img_path)
            if image is None:
                print(f"Skipping unreadable image: {img_name}")
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                # Extract coordinates (x, y, z) and flatten them
                landmarks = [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
                flattened_coordinates = np.array(landmarks).flatten().tolist()
                # Append the action label and save to CSV
                writer.writerow(flattened_coordinates + [action])
                file.flush()
            else:
                print(f"No landmarks detected in: {img_name}")

print(f"Pose coordinates and actions saved to {output_csv}")
