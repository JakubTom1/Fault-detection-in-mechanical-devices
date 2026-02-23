import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

# === PATHS ===
SCRIPT_PATH = os.getcwd()
INPUT_DIR = os.path.abspath(os.path.join(SCRIPT_PATH, '..', 'data', 'AnalysisData'))
OUTPUT_DIR = os.path.abspath(os.path.join(SCRIPT_PATH, '..', 'data', 'preprocessed', 'analysis_data'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === DATA LOADING ===
# Load and combine all CSV files from AnalysisData directory
dataframes = [
    pd.read_csv(os.path.join(INPUT_DIR, 'healthy.csv')).assign(Fault_Condition='healthy'),
    pd.read_csv(os.path.join(INPUT_DIR, 'healthy_zip.csv')).assign(Fault_Condition='healthy_zip'),
    pd.read_csv(os.path.join(INPUT_DIR, 'faulty.csv')).assign(Fault_Condition='faulty'),
    pd.read_csv(os.path.join(INPUT_DIR, 'faulty_zip.csv')).assign(Fault_Condition='faulty_zip')
]

data = pd.concat(dataframes, ignore_index=True)
print(f"Loaded {len(data)} records from {len(dataframes)} files.")

# === FEATURE SELECTION ===
# Select all numeric columns containing CURRENT or ROTO
feature_cols = [col for col in data.columns if any(x in col for x in ["CURRENT", "ROTO"])]
X = data[feature_cols]
y = data["Fault_Condition"]


# === TRAIN/VAL/TEST SPLIT ===
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
#X_val, X_test, y_val, y_test = train_test_split(
#    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
#)

# === MULTICOLLINEARITY REDUCTION ===
corr_matrix = X_train_raw.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find features with correlation greater than 0.90
threshold = 0.90
to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > threshold)]

print(f"Removing {len(to_drop)} highly correlated features: {to_drop}")
X_train_raw = X_train_raw.drop(columns=to_drop)
X_test_raw = X_test_raw.drop(columns=to_drop)
# Update feature_cols list so the scaler and saving steps use the correct columns
feature_cols = X_train_raw.columns.tolist()

# === STANDARDIZATION ===
scaler = StandardScaler()
X_train = pd.DataFrame(
    scaler.fit_transform(X_train_raw),
    columns=feature_cols,
    index=X_train_raw.index
)

X_test = pd.DataFrame(
    scaler.transform(X_test_raw),
    columns=feature_cols,
    index=X_test_raw.index
)

# === SAVE TO FILES ===
# Reset indices for proper concatenation
X_train_reset = X_train.reset_index(drop=True)
#X_val_reset = X_val.reset_index(drop=True)
X_test_reset = X_test.reset_index(drop=True)
y_train_reset = y_train.reset_index(drop=True)
#y_val_reset = y_val.reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)

train_df = pd.concat([X_train_reset, y_train_reset], axis=1)
#val_df = pd.concat([X_val_reset, y_val_reset], axis=1)
test_df = pd.concat([X_test_reset, y_test_reset], axis=1)

train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
#val_df.to_csv(os.path.join(OUTPUT_DIR, "val.csv"), index=False)
test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

# Save scaler and feature names for later use
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))
joblib.dump(feature_cols, os.path.join(OUTPUT_DIR, "feature_names.pkl"))

print("✅ Data saved to preprocessed directory")
print(f"Train: {len(train_df)} |  Test: {len(test_df)}") #Val: {len(val_df)} |
print(f"Train shape: {train_df.shape} |  Test shape: {test_df.shape}") # Val shape: {val_df.shape} |
print(f"Features: {len(feature_cols)}")
print(f"Class distribution:")
for class_name, count in y.value_counts().items():
    print(f"  {class_name}: {count}")
