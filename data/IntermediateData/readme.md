# Motor Condition Data Analysis

## Original Data Description
- **Sampling Frequency:** 50 kHz
- **Experiment Duration:** 20 seconds
- **Data Sources:** 
  - `healthy_1` - first batch of data from healty motor
  - `healthy_2` - second batch of data from healty motor
  - `faulty_1` - first batch of data from motor with electrical fault
  - `faulty_2` - second batch of data from motor with electrical fault
  - `healthy_zip_1` - first batch of data from motor with mechanical fault
  - `healthy_zip_2` - second batch of data from motor with mechanical fault
  - `faulty_zip_1` - first batch of data from motor with both electrical and mechanical fault
  - `faulty_zip_2` - second batch of data from motor with both electrical and mechanical fault

 
## Data Preprocessing Steps

### 1. Conversion from Voltage
#### Current Conversion
- Original: Voltage range 2.5V to 5V
  - 2.5V represents 0A
  - 5V represents 20A

- Conversion Formula: 
  ```
  Current (A) = (CURRENT (V) - 2.5) * 20 / 2.5
  ```

#### Rotational Speed Conversion
- Original: Voltage range 0V to 2V
  - 0V represents 0 RPM
  - 1V represents 10,000 RPM
  - 2V represents 20,000 RPM

- Conversion Formula:
  ```
  Rotational Speed (RPM) = ROTO (V) * 10,000
  ```

### 2. Windowing and Sampling
- **Window Size:** 0.8 seconds
- **Sampling Strategy:**
  - Start time adjusted for each dataset
  - Time reset relative to window start
  - Experiment IDs assigned based on window progression

### 3. Output Files
Generated files in this directory:
- `healthy.csv`: Healthy condition samples
- `healthy_zip.csv`: Healthy condition with ziptie samples
- `faulty.csv`: Faulty condition samples
- `faulty_zip.csv`: Faulty condition with ziptie samples


## Output Data Columns
- `Time (s)`: Time within each window (reset to 0 at window start)
- `Experiment ID`: Unique identifier for each window
- `CURRENT (A)`: Motor current in amperes
- `ROTO (RPM)`: Rotational speed in RPM

## Processing Notes
- Original voltage columns removed after conversion
- Data saved with reset time and consistent formatting

## Downsampling

We also used downsampling of our data for better visualization and more efficient processing. Specifically, each signal was downsampled by a factor of 10, significantly reducing the data size while preserving the overall structure of the signals.

This step was performed using custom MATLAB scripts, which automated the process across all datasets. The following scripts were used:

- `load_csv_matlab.m`
- `load_csv_save_mat.m`
- `load_csv.m`

These scripts handled tasks such as loading the raw signals, applying the downsampling operation, and saving the resulting data in a format suitable for further analysis and plotting. All used scripts are contained in `Scripts/ProcessingScripts` directory.
