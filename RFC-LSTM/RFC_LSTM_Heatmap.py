import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from keras._tf_keras.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# === 1. Load the Data ===
data = pd.read_csv(r"C:\Users\user\Downloads\All_pose_landmarks.csv")
X = data.iloc[:, :-1].values    # All features (pose landmarks)
y = data.iloc[:, -1].values     # Action labels

# === 2. Load the Preprocessing Objects ===
# Load the saved label encoder and scaler (from your training process)
label_encoder = joblib.load(r"D:\RFC_LSTM\label_encoder.pkl")
scaler = joblib.load(r"D:\RFC_LSTM\scaler.pkl")

# Encode labels using the loaded label encoder
y_encoded = label_encoder.transform(y)

# For RFC, use the original features.
# For LSTM, use the scaled features.
X_scaled = scaler.transform(X)

# === 3. Split the Data into Test Sets ===
# RFC was trained on raw (unscaled) features; split accordingly.
_, X_test_rfc, _, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# For LSTM, split the scaled data.
_, X_test_scaled, _, _ = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
# LSTM model expects input shape: (samples, timesteps, num_features)
# In your training, you reshaped using: X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
X_test_lstm = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

# === 4. Load the Trained Models ===
rf_model = joblib.load(r"D:\RFC_LSTM\best_random_forest_model.pkl")
lstm_model = load_model(r"D:\RFC_LSTM\best_lstm_model.h5")

# === 5. Get Predictions from Both Models ===
# RFC predictions: probabilities per class
rf_preds_test = rf_model.predict_proba(X_test_rfc)  # shape: (n_samples, n_classes)
# LSTM predictions: probabilities per class
lstm_preds_test = lstm_model.predict(X_test_lstm)     # shape: (n_samples, n_classes)

# === 6. Combine Predictions with Optimized Weights ===
# Use the optimized weights found during training (adjust these if needed)
optimized_rf_weight = 0.45
optimized_lstm_weight = 0.55

# Compute weighted average of predictions
combined_preds_test = (optimized_rf_weight * rf_preds_test) + (optimized_lstm_weight * lstm_preds_test)
final_preds_test = np.argmax(combined_preds_test, axis=1)

# === 7. Evaluation Metrics ===
accuracy = accuracy_score(y_test, final_preds_test)
f1 = f1_score(y_test, final_preds_test, average="macro")
print("Test Accuracy: {:.2f}".format(accuracy))
print("F1-Score (Macro): {:.4f}".format(f1))
print("\nClassification Report:")
print(classification_report(y_test, final_preds_test, target_names=label_encoder.classes_))

# === 8. Plot the Confusion Matrix Heatmap ===
cm = confusion_matrix(y_test, final_preds_test)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - RFC-LSTM Ensemble")
plt.show()
