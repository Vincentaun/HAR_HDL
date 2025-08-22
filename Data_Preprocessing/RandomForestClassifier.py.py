import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import joblib



# Load the CSV file
data = pd.read_csv(r"C:\Users\user\Downloads\RFC_FILE\All_pose_landmarks.csv")

# Separate features and labels
X = data.iloc[:, :-1].values  # All columns except the last
y = data.iloc[:, -1].values   # Last column is the action label

# Encode labels (if necessary)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
class_mapping = dict(zip(range(len(label_encoder.classes_)), label_encoder.classes_))
#print(class_mapping)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

# Initialize Random Forest
rf = RandomForestClassifier(random_state=42)
print("Here")
# Perform grid search
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=3)
grid_search.fit(X_train, y_train)

# Best parameters
print("Best Parameters:", grid_search.best_params_)

# Train final model with best parameters
best_rf = grid_search.best_estimator_
best_rf.fit(X_train, y_train)


# Plot feature importance
importance = best_rf.feature_importances_
plt.figure(figsize=(10, 6))
plt.bar(range(len(importance)), importance)
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.title("Feature Importance in Action Classification")
plt.show()


joblib.dump(best_rf, "random_forest_model.pkl")