from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, Conv1D, Flatten, Dropout
from keras._tf_keras.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


# Load the CSV file
data = pd.read_csv(r"C:\Users\User\Desktop\python\113-1 AI Practice\H36M\pose_coordinates.csv")

# Separate features and labels
X = data.iloc[:, :-1].values  # All columns except the last
y = data.iloc[:, -1].values   # Last column is the action label

# Encode labels (if necessary)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
#class_mapping = dict(zip(range(len(label_encoder.classes_)), label_encoder.classes_))


# Prepare data
y_one_hot = to_categorical(y_encoded)  # One-hot encode labels
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

# Reshape data for Conv1D
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)  # (samples, features, channels)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Define CNN model
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    Dropout(0.3),
    Conv1D(filters=128, kernel_size=3, activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(y_one_hot.shape[1], activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)
# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)



# Plot accuracy and loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Over Epochs')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Over Epochs')

plt.show()



# Save the model
model.save("pose_action_model.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open("cnn_pose_action_model.tflite", "wb") as f:
    f.write(tflite_model)
