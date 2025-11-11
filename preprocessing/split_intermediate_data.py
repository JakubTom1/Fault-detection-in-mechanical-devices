import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib  # do zapisu skalera

# === PATHS ===
SCRIPT_PATH = os.getcwd()
INPUT_DIR = os.path.abspath(os.path.join(SCRIPT_PATH, '..', 'data', 'IntermediateData'))
OUTPUT_DIR = os.path.join(SCRIPT_PATH, '../data', 'preprocessed', 'neural_network')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === DATA LOADING ===
# Load and combine CSV files, each with a Fault_Condition label
dataframes = [
    pd.read_csv(os.path.join(INPUT_DIR, 'healthy.csv')).assign(Fault_Condition='healthy'),
    pd.read_csv(os.path.join(INPUT_DIR, 'healthy_zip.csv')).assign(Fault_Condition='healthy_zip'),
    pd.read_csv(os.path.join(INPUT_DIR, 'faulty.csv')).assign(Fault_Condition='faulty'),
    pd.read_csv(os.path.join(INPUT_DIR, 'faulty_zip.csv')).assign(Fault_Condition='faulty_zip')
]

data = pd.concat(dataframes, ignore_index=True)

# Ensure columns are in correct order
data = data[['Time (s)', 'CURRENT (A)', 'ROTO (RPM)', 'Experiment ID', 'Fault_Condition']]

# Map each experiment to its label
experiment_labels = data.groupby('Experiment ID')['Fault_Condition'].first()
experiment_ids = experiment_labels.index.values

# === TRAIN/VAL/TEST SPLIT ===
# Split by experiment, not by rows
train_ids, temp_ids = train_test_split(
    experiment_ids,
    test_size=0.3,
    stratify=experiment_labels[experiment_ids],
    random_state=42
)
val_ids, test_ids = train_test_split(
    temp_ids,
    test_size=0.5,
    stratify=experiment_labels[temp_ids],
    random_state=42
)

# Filter full data by experiment IDs
train_data = data[data['Experiment ID'].isin(train_ids)].copy()
val_data = data[data['Experiment ID'].isin(val_ids)].copy()
test_data = data[data['Experiment ID'].isin(test_ids)].copy()

# === STANDARDIZATION ===
scaler = StandardScaler()

# Fit only on training data
scaler.fit(train_data[['CURRENT (A)', 'ROTO (RPM)']])

# Apply the same transformation to all sets
train_data[['CURRENT (A)', 'ROTO (RPM)']] = scaler.transform(train_data[['CURRENT (A)', 'ROTO (RPM)']])
val_data[['CURRENT (A)', 'ROTO (RPM)']] = scaler.transform(val_data[['CURRENT (A)', 'ROTO (RPM)']])
test_data[['CURRENT (A)', 'ROTO (RPM)']] = scaler.transform(test_data[['CURRENT (A)', 'ROTO (RPM)']])


# === SAVE TO FILES ===
# Save scaled datasets
train_data.to_csv(os.path.join(OUTPUT_DIR, 'train.csv'), index=False)
val_data.to_csv(os.path.join(OUTPUT_DIR, 'val.csv'), index=False)
test_data.to_csv(os.path.join(OUTPUT_DIR, 'test.csv'), index=False)

# Save scaler
scaler_path = os.path.join(OUTPUT_DIR, 'scaler.pkl')
joblib.dump(scaler, scaler_path)

# --- VERIFY ---
print(f"Train experiments: {len(train_ids)}, Validation: {len(val_ids)}, Test: {len(test_ids)}")
print(f"Train rows: {len(train_data)}, Val rows: {len(val_data)}, Test rows: {len(test_data)}\n")

print("Fault condition distribution by experiments:")
print("Train:\n", experiment_labels[train_ids].value_counts(normalize=True))
print("Val:\n", experiment_labels[val_ids].value_counts(normalize=True))
print("Test:\n", experiment_labels[test_ids].value_counts(normalize=True))

print("\nScaler saved to:", scaler_path)
print("Mean used for scaling:\n", dict(zip(['CURRENT (A)', 'ROTO (RPM)'], scaler.mean_)))
print("Standard deviations used for scaling::\n", dict(zip(['CURRENT (A)', 'ROTO (RPM)'], scaler.scale_)))
