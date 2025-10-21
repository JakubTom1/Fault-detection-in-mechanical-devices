import pandas as pd
import os
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# === PATHS ===
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_PATH, '..', 'data', 'preprocessed', 'decision_tree'))
MODELS_DIR = os.path.abspath(os.path.join(SCRIPT_PATH, '..', 'models','decision_tree'))
FEATURES_NAMES_DIR = os.path.abspath(os.path.join(DATA_DIR,'feature_names.pkl'))
os.makedirs(MODELS_DIR, exist_ok=True)

# Load preprocessed data
train_data = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
val_data = pd.read_csv(os.path.join(DATA_DIR, 'val.csv'))
test_data = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

# Load feature names from saved file
feature_cols = joblib.load(FEATURES_NAMES_DIR)

# Prepare features and labels
X_train = train_data[feature_cols]
y_train = train_data['Fault_Condition']
X_val = val_data[feature_cols]
y_val = val_data['Fault_Condition']
X_test = test_data[feature_cols]
y_test = test_data['Fault_Condition']

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate on validation set
val_pred = rf_model.predict(X_val)
val_accuracy = accuracy_score(y_val, val_pred)
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Evaluate on test set
test_pred = rf_model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")
print("\nTest Classification Report:")
print(classification_report(y_test, test_pred))

# Save the trained model
model_path = os.path.join(MODELS_DIR, 'random_forest.pkl')
joblib.dump(rf_model, model_path)
print(f"\nModel saved to: {model_path}")