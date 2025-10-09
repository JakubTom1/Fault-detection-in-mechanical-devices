import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Define paths
script_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(script_dir, 'data', 'IntermediateData')
output_dir = os.path.join(script_dir, 'data', 'preprocessed')

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load and combine CSV files with Fault_Condition column
dataframes = [
    pd.read_csv(os.path.join(input_dir, 'healthy.csv')).assign(Fault_Condition='healthy'),
    pd.read_csv(os.path.join(input_dir, 'healthy_zip.csv')).assign(Fault_Condition='healthy_zip'),
    pd.read_csv(os.path.join(input_dir, 'faulty.csv')).assign(Fault_Condition='faulty'),
    pd.read_csv(os.path.join(input_dir, 'faulty_zip.csv')).assign(Fault_Condition='faulty_zip')
]
data = pd.concat(dataframes, ignore_index=True)

# Ensure columns are in the desired order
data = data[['Time (s)', 'CURRENT (A)', 'ROTO (RPM)', 'Experiment ID', 'Fault_Condition']]

# Get unique Experiment IDs and their corresponding Fault_Condition
experiment_ids = data['Experiment ID'].unique()
labels = data.groupby('Experiment ID')['Fault_Condition'].first()

# Perform stratified split by Experiment ID (70% train, 15% val, 15% test)
train_ids, temp_ids = train_test_split(
    experiment_ids,
    test_size=0.3,  # 30% for val + test
    stratify=labels[experiment_ids],
    random_state=42
)
val_ids, test_ids = train_test_split(
    temp_ids,
    test_size=0.5,  # Split remaining 30% into 15% val, 15% test
    stratify=labels[temp_ids],
    random_state=42
)

# Create training, validation, and test datasets
train_data = data[data['Experiment ID'].isin(train_ids)]
val_data = data[data['Experiment ID'].isin(val_ids)]
test_data = data[data['Experiment ID'].isin(test_ids)]

# Save datasets to data/preprocessed
train_data.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
val_data.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
test_data.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

# Verify the split proportions
print(f"Training set size: {len(train_data)} rows, {len(train_data['Experiment ID'].unique())} experiments")
print(f"Validation set size: {len(val_data)} rows, {len(val_data['Experiment ID'].unique())} experiments")
print(f"Test set size: {len(test_data)} rows, {len(test_data['Experiment ID'].unique())} experiments")

# Verify fault condition distribution
print("\nFault condition distribution:")
print("Training set:\n", train_data['Fault_Condition'].value_counts(normalize=True))
print("Validation set:\n", val_data['Fault_Condition'].value_counts(normalize=True))
print("Test set:\n", test_data['Fault_Condition'].value_counts(normalize=True))