# Fault-detection-in-mechanical-devices
This repository contains the implementation of statistical and machine learning models for fault detection in mechanical devices based on selected signals such as current, power or vibration.   It has been developed as part of a thesis project.

## 🎯 Project Goal
The main aim of this work is to **develop and evaluate statistical models for detecting faults in mechanical devices** using selected sensor signals (e.g., current, rpm).  
The project provides a **functional application** that demonstrates the performance of the selected algorithms and allows their direct comparison.

## 📋 Scope of Work
As part of the thesis, the following tasks are covered:
- **Data selection and preparation** – collecting and preprocessing relevant signals.
- **Review and selection of statistical models** – evaluating various approaches to fault detection.
- **Algorithm design and implementation** – building models such as decision trees, neural networks, and classical statistical models.
- **Testing and evaluation** – producing metrics, reports, and visualizations for model performance.

```
project_root/
│
├── data/ # raw and processed data
│ ├── raw/ # original CSV or sensor data
│ ├── processed/ # saved preprocessed data
│
├── models_code/ # all model implementation files (.py)
│ ├── neural_network.py # neural network model definition/training
│ ├── decision_tree.py # decision tree model definition/training
│ ├── statistical_model.py # statistical model definition/training
│
├── preprocessing/ # data loading & preparation
│ ├── preprocessing.py # data cleaning and splitting
│
├── tests/ # Python test scripts for models & preprocessing
│ ├── test_neural_network.py
│ ├── test_decision_tree.py
│ ├── test_statistical_model.py
│ ├── test_preprocessing.py
│
├── evaluation/ # evaluation and comparison
│ ├── evaluate.py # common metrics & comparison
│
├── models/ # saved trained models (pkl)
├── results/ # evaluation results, plots, reports
│
├── requirements.txt # list of Python dependencies
└── main.py # entry point to run training and evaluation
```