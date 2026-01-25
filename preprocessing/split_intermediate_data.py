import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib  # do zapisu skalera

WINDOW_SIZE = 100
STEP = 50

# === PATHS ===
SCRIPT_PATH = os.getcwd()
INPUT_DIR = os.path.abspath(os.path.join(SCRIPT_PATH, '..', 'data', 'IntermediateData'))
OUTPUT_DIR = os.path.join(SCRIPT_PATH, '../data', 'preprocessed', 'time_window_data')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === DATA LOADING ===
dataframes = []

for idx, (file_name, fault_cond) in enumerate([('healthy.csv', 'healthy'),
                                                 ('healthy_zip.csv', 'healthy_zip'),
                                                 ('faulty.csv', 'faulty'),
                                                 ('faulty_zip.csv', 'faulty_zip')]):
    df = pd.read_csv(os.path.join(INPUT_DIR, file_name))
    df['Experiment ID'] = df['Experiment ID'] + idx * 1000
    df['Fault_Condition'] = fault_cond
    dataframes.append(df)

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

# === LABEL ENCODING ===
label_encoder = LabelEncoder()
all_labels = data['Fault_Condition'].unique()
label_encoder.fit(all_labels)

def create_sequences(df, window_size, step, label_encoder):
    X = []
    y = []
    for exp_id in df['Experiment ID'].unique():
        exp_data = df[df['Experiment ID'] == exp_id].sort_values('Time (s)')
        features = exp_data[['CURRENT (A)', 'ROTO (RPM)']].values  
        label = exp_data['Fault_Condition'].iloc[0] 

        for i in range(0, len(features) - window_size + 1, step):
            X.append(features[i:i + window_size])
            y.append(label_encoder.transform([label])[0])

    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_data, WINDOW_SIZE, STEP, label_encoder)
X_val, y_val = create_sequences(val_data, WINDOW_SIZE, STEP, label_encoder)
X_test, y_test = create_sequences(test_data, WINDOW_SIZE, STEP, label_encoder)

# === SAVE TO FILES ===
# Save scaled datasets
train_data.to_csv(os.path.join(OUTPUT_DIR, 'train.csv'), index=False)
val_data.to_csv(os.path.join(OUTPUT_DIR, 'val.csv'), index=False)
test_data.to_csv(os.path.join(OUTPUT_DIR, 'test.csv'), index=False)

np.savez(os.path.join(OUTPUT_DIR, 'train.npz'), X=X_train, y=y_train)
np.savez(os.path.join(OUTPUT_DIR, 'val.npz'), X=X_val, y=y_val)
np.savez(os.path.join(OUTPUT_DIR, 'test.npz'), X=X_test, y=y_test)

# Save scaler
scaler_path = os.path.join(OUTPUT_DIR, 'scaler.pkl')
joblib.dump(scaler, scaler_path)

label_encoder_path = os.path.join(OUTPUT_DIR, 'label_encoder.pkl')
joblib.dump(label_encoder, label_encoder_path)

# --- VERIFY ---
print(f"Train experiments: {len(train_ids)}, Validation: {len(val_ids)}, Test: {len(test_ids)}")
print(f"Train sequences: {len(X_train)}, Val sequences: {len(X_val)}, Test sequences: {len(X_test)}\n")
print(f"Shape of X_train: {X_train.shape} (samples, timesteps, features)")
print(f"Shape of y_train: {y_train.shape}\n")

print("Fault condition distribution by experiments:")
print("Train:\n", experiment_labels[train_ids].value_counts(normalize=True))
print("Val:\n", experiment_labels[val_ids].value_counts(normalize=True))
print("Test:\n", experiment_labels[test_ids].value_counts(normalize=True))

print("\nScaler saved to:", scaler_path)
print("Mean used for scaling:\n", dict(zip(['CURRENT (A)', 'ROTO (RPM)'], scaler.mean_)))
print("Standard deviations used for scaling:\n", dict(zip(['CURRENT (A)', 'ROTO (RPM)'], scaler.scale_)))

print("\nLabel encoder saved to:", label_encoder_path)
print("Label classes:\n", dict(enumerate(label_encoder.classes_)))