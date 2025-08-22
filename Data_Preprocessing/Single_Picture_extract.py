import cv2
import mediapipe as mp
import numpy as np
import time


# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
mp_drawing = mp.solutions.drawing_utils

# Load the image
image_path = r"D:\S11\S11_Phoning_2.55011271_002156.jpg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the image to detect pose landmarks
results = pose.process(image_rgb)

# Check if landmarks were detected
if results.pose_landmarks:
    landmarks = results.pose_landmarks.landmark
    # Extract coordinates as (x, y, z) values
    coordinates = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    print("Coordinates:", coordinates)

    # Optionally visualize the landmarks
    annotated_image = image.copy()
    mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.imshow("Pose Landmarks", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the coordinates to a file
    np.savetxt("pose_coordinates.csv", coordinates, delimiter=",", header="x,y,z", comments="")
else:
    print("No pose landmarks detected.")
