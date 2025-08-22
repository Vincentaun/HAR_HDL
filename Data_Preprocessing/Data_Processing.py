import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def normalize_pose(data, reference_joint_index=0):
    """
    Normalize pose data by centering it around the reference joint.
    Args:
        data: Pose data of shape (num_frames, 3, num_joints)
        reference_joint_index: Index of the reference joint (e.g., 0 for the hip).
    Returns:
        Normalized pose data of the same shape.
    """
    # Extract the reference joint coordinates
    reference_joint = data[:, :, reference_joint_index]  # Shape: (num_frames, 3)
    
    # Center all joints relative to the reference joint
    normalized_data = data - reference_joint[:, :, None]  # Subtract reference joint from all joints
    return normalized_data


def plot_pose(pose, ax=None):
    """
    Plot a single 3D pose.
    Args:
        pose: Pose data of shape (3, num_joints)
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    
    x, y, z = pose
    ax.scatter(x, y, z, c='r', marker='o')
    
    # Optionally connect the joints (update based on your skeleton structure)
    skeleton = [
        (0, 1), (1, 2), (2, 3),  # Example connections
        (0, 4), (4, 5), (5, 6)   # Add more as needed
    ]
    for start, end in skeleton:
        ax.plot([x[start], x[end]], [y[start], y[end]], [z[start], z[end]], 'b')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()



# Step 1: Read the text file
file_path = r"D:\h36m_annot\h36m\annot\train_images.txt"
with open(file_path, 'r') as f:
    lines = f.readlines()

# Step 2: Extract filenames and activity labels
image_files = []
activity_labels = []

for line in lines:
    filename = line.strip()  # Remove newline character
    activity = filename.split('_')[1]  # Extract activity (e.g., "Discussion", "Eating")
    image_files.append(filename)
    activity_labels.append(activity)

# Step 3: Map activity names to numeric labels
unique_activities = sorted(set(activity_labels))  # Unique activities
activity_to_label = {activity: idx for idx, activity in enumerate(unique_activities)}
numeric_labels = [activity_to_label[activity] for activity in activity_labels]

# Step 4: Split into training and validation sets
train_files, val_files, train_labels, val_labels = train_test_split(
    image_files, numeric_labels, test_size=0.2, random_state=42
)

# Step 5: Output the results
print("Unique Activities:", unique_activities)
print("Train Files (first 5):", train_files[:5])
print("Train Labels (first 5):", train_labels[:5])
print("Validation Files (first 5):", val_files[:5])
print("Validation Labels (first 5):", val_labels[:5])







# Path to the dataset
#dataset_path = "/path/to/H3.6M/poses"

# Load a sample .mat file (replace with the actual file)
pose_data = loadmat(r"D:\h36m_annot\h36m\annot\valid.mat")

# Check the structure of the file
#print(pose_data.keys())

# Example: Extract poses and activities
poses = pose_data["annot"]  # 3D pose landmarks (x, y, z)

#print(# Inspect the structure of the 'annot' keytype(pose_data['annot']), pose_data['annot'].dtype, pose_data['annot'].shape)

# Extract the fields within the 'annot' array
annot = pose_data['annot']
annot_fields = annot[0, 0].dtype.fields
print(annot_fields.keys())  # List all fields within 'annot'

# Inspect the structure of the 'S' field (3D pose data)
S_data = annot[0, 0]['S']
print(type(S_data), S_data.shape, S_data[0]) if len(S_data) > 0 else print("Empty")


annot_contents = annot[0, 0]
# Extracting the activity labels (assumed to be in the first column of the first component of the array)
activity_labels = annot_contents[0]  # Accessing the first component of 'annot_contents'
# Step 1: Extract Activity Names
activity_names = [str(str(label[0]).split('_')[1]).split('.')[0] for label in activity_labels]  # Extract the activity portion
#print(str(activity_labels[0]).split("_")[1])
# # Step 2: Encode Activity Names into Numeric Labels
unique_activities = sorted(set(activity_names))  # Get unique activity names
activity_to_label = {activity: idx for idx, activity in enumerate(unique_activities)}  # Map activities to integers
numeric_labels = np.array([activity_to_label[name] for name in activity_names])

print((activity_to_label))

# # Finding unique activity names
# unique_activities = set(activity_names)
# unique_activities




# Normalize the data (assuming joint 0 is the hip)
normalized_data = normalize_pose(np.array(S_data))
#print("Normalized Data Shape:", normalized_data.shape)

# Split into train (70%), validation (15%), and test (15%)
train_data, test_data = train_test_split(normalized_data, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)

# print("Train Data Shape:", train_data.shape)
# print("Validation Data Shape:", val_data.shape)
# print("Test Data Shape:", test_data.shape)

# Plot the first pose in the dataset
#plot_pose(normalized_data[0])

# print(type(test_labels))
#numeric_labels = np.dtype()

# # Split labels accordingly
# train_labels, test_labels = train_test_split(numeric_labels, test_size=0.3, random_state=42)
# val_labels, test_labels = train_test_split(test_labels, test_size=0.5, random_state=42)

# Flatten each pose into a 1D array (if required by the model)
X_train = train_data.reshape(train_data.shape[0], -1)
X_val = val_data.reshape(val_data.shape[0], -1)
X_test = test_data.reshape(test_data.shape[0], -1)

print("Flattened Train Data Shape:", X_train.shape)

train_labels = numeric_labels
print(train_labels.shape)
# Train a model (assuming `train_labels` and `val_labels` are defined)
clf = RandomForestClassifier()
clf.fit(X_train, train_labels)

# Validate the model
val_predictions = clf.predict(X_val)
print("Validation Accuracy:", accuracy_score(val_labels, val_predictions))


#activities = pose_data["activities"]  # Corresponding activity labels
