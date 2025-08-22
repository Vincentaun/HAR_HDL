import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import LSTM, Dense, Dropout
from keras._tf_keras.keras.utils import to_categorical
from keras._tf_keras.keras.callbacks import EarlyStopping
import keras_tuner as kt
from scipy.optimize import minimize
import joblib
import sys

sys.stdout = open("RFC_LSTM_log.txt", mode='a')
# --- Load and Preprocess Data ---
data = pd.read_csv(r"C:\Users\user\Downloads\RFC-LSTM\All_pose_landmarks.csv")

# Separate features and labels
X = data.iloc[:, :-1].values  # All columns except the last
y = data.iloc[:, -1].values   # Last column is the action label

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
class_mapping = dict(zip(range(len(label_encoder.classes_)), label_encoder.classes_))
print("Class Mapping:", class_mapping)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# --- Random Forest Hyperparameter Search ---
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best RFC Model
best_rf = grid_search.best_estimator_
print("Best RFC Parameters:", grid_search.best_params_)

# Save the RFC Model
joblib.dump(best_rf, "best_random_forest_model.pkl")

# RFC Predictions
rf_preds_train = best_rf.predict_proba(X_train)
rf_preds_test = best_rf.predict_proba(X_test)

# --- LSTM Hyperparameter Search ---
# Normalize data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

# One-hot encode labels
y_one_hot = to_categorical(y_encoded)

# Split data
X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(
    X_reshaped, y_one_hot, test_size=0.2, random_state=42
)

# Define LSTM Hypermodel
def build_lstm_model(hp):
    model = Sequential()
    model.add(
        LSTM(
            units=hp.Int("units", min_value=64, max_value=256, step=32),
            input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]),
            return_sequences=False
        )
    )
    model.add(Dropout(hp.Float("dropout", 0.1, 0.5, step=0.1)))
    model.add(Dense(
        units=hp.Int("dense_units", min_value=32, max_value=128, step=16),
        activation="relu"
    ))
    model.add(Dropout(hp.Float("dropout_dense", 0.1, 0.5, step=0.1)))
    model.add(Dense(y_one_hot.shape[1], activation="softmax"))
    model.compile(
        optimizer=hp.Choice("optimizer", ["adam", "rmsprop"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# Initialize Keras Tuner
tuner = kt.Hyperband(
    build_lstm_model,
    objective="val_accuracy",
    max_epochs=50,
    factor=3,
    directory="hyperband_lstm",
    project_name="h36m_lstm_tuning"
)

# Search for Best LSTM Hyperparameters
tuner.search(
    X_train_lstm, y_train_lstm,
    validation_data=(X_test_lstm, y_test_lstm),
    epochs=50,
    batch_size=32,
    callbacks=[EarlyStopping(monitor="val_loss", patience=5)]
)

# Get Best Hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best LSTM Hyperparameters:", best_hps.values)

# Train Best LSTM Model
best_lstm_model = tuner.hypermodel.build(best_hps)
best_lstm_model.fit(
    X_train_lstm, y_train_lstm,
    validation_data=(X_test_lstm, y_test_lstm),
    epochs=50,
    batch_size=32,
    callbacks=[EarlyStopping(monitor="val_loss", patience=5)]
)

# Save the LSTM Model
best_lstm_model.save("best_lstm_model.h5")

# LSTM Predictions
lstm_preds_train = best_lstm_model.predict(X_train_lstm)
lstm_preds_test = best_lstm_model.predict(X_test_lstm)

# --- Weight Optimization for Ensemble ---
def optimize_weights(weights):
    rf_weight, lstm_weight = weights
    rf_weight = rf_weight / (rf_weight + lstm_weight)
    lstm_weight = lstm_weight / (rf_weight + lstm_weight)
    combined_preds_test = (rf_weight * rf_preds_test) + (lstm_weight * lstm_preds_test)
    final_preds_test = np.argmax(combined_preds_test, axis=1)
    accuracy = accuracy_score(y_test, final_preds_test)
    return -accuracy

# Initial Guess for Weights
initial_weights = [0.5, 0.5]

# Optimize Weights
result = minimize(optimize_weights, initial_weights, method='Nelder-Mead', bounds=[(0, 1), (0, 1)])
optimized_rf_weight, optimized_lstm_weight = result.x
optimized_rf_weight /= (optimized_rf_weight + optimized_lstm_weight)
optimized_lstm_weight /= (optimized_rf_weight + optimized_lstm_weight)

print(f"Optimized RF Weight: {optimized_rf_weight:.2f}")
print(f"Optimized LSTM Weight: {optimized_lstm_weight:.2f}")

# --- Final Evaluation ---
combined_preds_test = (optimized_rf_weight * rf_preds_test) + (optimized_lstm_weight * lstm_preds_test)
final_preds_test = np.argmax(combined_preds_test, axis=1)
final_accuracy = accuracy_score(y_test, final_preds_test)

print(f"Final Test Accuracy with Optimized Weights: {final_accuracy:.2f}")

# Save Label Encoder and Scaler
joblib.dump(label_encoder, "label_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")
