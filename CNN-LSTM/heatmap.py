import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from keras._tf_keras.keras.utils import to_categorical
from keras._tf_keras.keras.models import load_model, Sequential
from sklearn.model_selection import train_test_split


# === 1. Load and Preprocess Test Data ===
# Load the CSV file containing your pose landmarks and action labels.
data = pd.read_csv(r"C:\Users\user\Downloads\All_pose_landmarks.csv")

# Separate features (X) and labels (y)
X = data.iloc[:, :-1].values  # All columns except the last
y = data.iloc[:, -1].values   # Last column contains the action label

# Encode labels into integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# One-hot encode labels for evaluation if needed
y_one_hot = to_categorical(y_encoded)

# Scale features using MinMaxScaler (using the same approach as in training)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Create sequences from the scaled data.
# For example, using a sequence length of 10 frames.
sequence_length = 10
X_sequences = []
y_sequences = []

for i in range(len(X_scaled) - sequence_length + 1):
    X_sequences.append(X_scaled[i : i + sequence_length])
    # Label for the sequence is taken as the label of the last frame in the sequence.
    y_sequences.append(y_encoded[i + sequence_length - 1])

X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)
y_sequences_one_hot = to_categorical(y_sequences)

# Reshape sequences for model input.
# Final shape: (num_samples, timesteps, num_features, channels)
# Here, channels is set to 1.
X_sequences = X_sequences.reshape(X_sequences.shape[0],
                                  X_sequences.shape[1],
                                  X_sequences.shape[2],
                                  1)

# Optionally, split data into train and test sets.
# Here, we use a simple train-test split, focusing on evaluation.
_, X_test, _, y_test = train_test_split(X_sequences, y_sequences_one_hot, test_size=0.2, random_state=42)

# === 2. Load the Final Model ===
# Load your saved CNN-LSTM model.
model = load_model("final_cnn_lstm_model_2.h5")

# === 3. Make Predictions on Test Data ===
# Get model predictions (probability distributions)
y_pred_probs = model.predict(X_test)
# Convert probabilities to class indices
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# === 4. Compute and Display the Confusion Matrix Heatmap ===
# Calculate the confusion matrix using scikit-learn.
cm = confusion_matrix(y_true, y_pred)

# Plot the confusion matrix as a heatmap.
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# === 5. Compute the F1-Score and Classification Report ===
# Compute the macro-average F1-score (each class contributes equally)
f1 = f1_score(y_true, y_pred, average='macro')
print("F1-Score (Macro): {:.4f}".format(f1))

# Print a detailed classification report (precision, recall, F1-score per class)
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))
