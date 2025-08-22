import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib


# Load the CSV file
data = pd.read_csv(r"C:\Users\User\Desktop\python\113-1 AI Practice\H36M\pose_coordinates.csv")

# Separate features and labels
X = data.iloc[:, :-1].values  # All columns except the last
y = data.iloc[:, -1].values   # Last column is the action label

# Encode labels (if necessary)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)


rf_model = joblib.load("random_forest_model.pkl")
y_pred = rf_model.predict(X_test)
print("Accuracy: ",accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred,target_names=label_encoder.classes_))