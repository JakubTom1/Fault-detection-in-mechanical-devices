# Experimental Data Description  

## Experiment Details  
- **Sampling Frequency (fs):** 50 kHz  
- **Experiment Duration:** 20 seconds  

Measurements come from 4 types of motors:
- healthy
- with mechanical damage
- with electrcail damage
- with both electrical and mechanical damage
Experiment on each motor was conduted twice under the same conditions.

## File Naming Convention  
Files are named as follows: 
(healthy/faulty)_(zip)_(experiment_number)

Example filenames:  
- `healthy_zip_2`  
- `faulty_1`  

## Data Structure  

| Column | Description |
|--------|-------------|
| **Sample** | Sample number |
| **Time** | Time in milliseconds |

### Analog Channels  

#### 1. CURRENT (V) – Total Motor Current  
| Voltage (V) | Current (A) |
|-------------|------------|
| 2.5V | 0A |
| 5V | 20A |

#### 2. ROTO (V) – Rotational Speed  
| Voltage (V) | Speed (RPM) |
|-------------|------------|
| 0V | 0 RPM |
| 1V | 10,000 RPM |
| 2V | 20,000 RPM |

> **Note:** Other channels are unconnected and can be ignored.  