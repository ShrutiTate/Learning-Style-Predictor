# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
df = pd.read_csv("learning_style_data.csv")

# Fill missing values with mode
df_filled = df.fillna(df.mode().iloc[0])

# Encode categorical variables
label_encoders = {}
for col in df_filled.columns:
    le = LabelEncoder()
    df_filled[col] = le.fit_transform(df_filled[col])
    label_encoders[col] = le

# Split features and target
X = df_filled.drop("Learning_Style", axis=1)
y = df_filled["Learning_Style"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save model and encoders
joblib.dump(model, "model.pkl")
joblib.dump(label_encoders, "encoders.pkl")
