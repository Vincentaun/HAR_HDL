import os
import shutil

# Paths
source_folders = [r"D:\S1", r"D:\S5", r"D:\S6", r"D:\S7", r"D:\S8", r"D:\S9", r"D:\S11"]  # Replace with your folder paths
destination_folder = r"D:\Training_Material"  # Replace with the path where you want Training_Material

# Create destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Copy images from each folder
for folder in source_folders:
    for file_name in os.listdir(folder):
        source_file = os.path.join(folder, file_name)
        if os.path.isfile(source_file):
            shutil.copy(source_file, destination_folder)

print(f"All images merged into {destination_folder}")
