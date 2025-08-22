import optuna
import tensorflow as tf
from keras._tf_keras.keras import Model, Sequential
from keras._tf_keras.keras.layers import Dense, LSTM, Flatten, Dropout, TimeDistributed, GlobalAveragePooling2D, Input, Conv1D
from keras._tf_keras.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras._tf_keras.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import sys
import pickle

sys.stdout = open('CNN_LSTM_Log2.txt', 'w')

# --- Load and Prepare Data ---
data = pd.read_csv(r"C:\Users\user\Downloads\All_pose_landmarks.csv")
# Separate features and labels
X = data.iloc[:, :-1].values  # All columns except the last
y = data.iloc[:, -1].values   # Action labels

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_one_hot = to_categorical(y_encoded)

# Rescale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Create sequences (e.g., 10 frames per sequence)
sequence_length = 10
X_sequences = []
y_sequences = []
for i in range(len(X_scaled) - sequence_length + 1):
    X_sequences.append(X_scaled[i:i + sequence_length])
    y_sequences.append(y_encoded[i + sequence_length - 1])
X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)
y_sequences_one_hot = to_categorical(y_sequences)

# Reshape to add a feature channel for Conv1D
# Final shape: (num_samples, timesteps, features, channels)
X_sequences = X_sequences.reshape(X_sequences.shape[0], X_sequences.shape[1], X_sequences.shape[2], 1)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_sequences, y_sequences_one_hot, test_size=0.2, random_state=42
)

# Extract dimensions for clarity
num_frames = X_train.shape[1]       # Number of timesteps per sequence
feature_dim = X_train.shape[2]      # Number of features per timestep
num_classes = y_train.shape[1]      # Number of classes

# --- Define Objective Function for Optuna ---
def objective(trial): 
    model = Sequential([
        # TimeDistributed Conv1D to process each frame's features
        TimeDistributed(Conv1D(
            filters=trial.suggest_int('filters', 32, 64, step=16),
            kernel_size=trial.suggest_categorical('kernel_size', [3, 5]),
            activation='relu'
        ), input_shape=(num_frames, feature_dim, 1)),
        TimeDistributed(Dropout(trial.suggest_float('cnn_dropout', 0.1, 0.3, step=0.1))),
        TimeDistributed(Conv1D(
            filters=trial.suggest_int('filters_2', 64, 128, step=32),
            kernel_size=trial.suggest_categorical('kernel_size_2', [3, 5]),
            activation='relu'
        )),
        TimeDistributed(Flatten()),
        LSTM(trial.suggest_int('lstm_units', 64, 128, step=32), activation='tanh'),
        Dropout(trial.suggest_float('lstm_dropout', 0.1, 0.5, step=0.1)),
        Dense(trial.suggest_int('dense_units', 64, 128, step=32), activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
    
    model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=20,  # Fewer epochs during tuning
        batch_size=trial.suggest_categorical('batch_size', [16, 32]),
        callbacks=[early_stopping, lr_scheduler],
        verbose=1
    )
    
    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
    return accuracy

# --- Optimize Hyperparameters with Optuna ---
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

print("Best trial:")
trial = study.best_trial
print("  Value: {:.4f}".format(trial.value))
print("  Params:")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# --- Build the Final Model with Best Hyperparameters ---
best_params = trial.params

final_model = Sequential()
final_model.add(TimeDistributed(Conv1D(
    filters=best_params['filters'],
    kernel_size=best_params['kernel_size'],
    activation='relu'
), input_shape=(num_frames, feature_dim, 1)))
final_model.add(TimeDistributed(Dropout(best_params['cnn_dropout'])))
final_model.add(TimeDistributed(Conv1D(
    filters=best_params['filters_2'],
    kernel_size=best_params['kernel_size_2'],
    activation='relu'
)))
final_model.add(TimeDistributed(Flatten()))
final_model.add(LSTM(best_params['lstm_units'], activation='tanh'))
final_model.add(Dropout(best_params['lstm_dropout']))
final_model.add(Dense(best_params['dense_units'], activation='relu'))
final_model.add(Dense(num_classes, activation='softmax'))

final_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping_final = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler_final = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)

final_history = final_model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=best_params['batch_size'],
    callbacks=[early_stopping_final, lr_scheduler_final],
    verbose=1
)

loss, accuracy = final_model.evaluate(X_test, y_test, verbose=1)
print(f"Final Model Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Save the final model
final_model.save("final_cnn_lstm_model_2.h5")

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("label_encoder.pkl", "wb") as f :
    pickle.dump(label_encoder, f)

sys.stdout.close()
sys.stdout = sys.__stdout__