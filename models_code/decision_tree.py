import pandas as pd
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Define paths
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(script_dir, 'data', 'preprocessed')
models_dir = os.path.join(script_dir, 'models')

# Create models directory if it doesn't exist
os.makedirs(models_dir, exist_ok=True)

# Load preprocessed data
train_data = pd.read_csv(os.path.join(data_dir, 'train.csv'))
val_data = pd.read_csv(os.path.join(data_dir, 'val.csv'))
test_data = pd.read_csv(os.path.join(data_dir, 'test.csv'))

# Prepare features and labels
feature_cols = ['Time (s)', 'CURRENT (A)', 'ROTO (RPM)']
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
model_path = os.path.join(models_dir, 'random_forest.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(rf_model, f)
print(f"\nModel saved to: {model_path}")