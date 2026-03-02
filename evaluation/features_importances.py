import pandas as pd
import joblib
from sklearn.inspection import permutation_importance
import os
import xarray as xr
from sklearn.preprocessing import LabelEncoder
import numpy as np

# === PATHS ===
SCRIPT_PATH = os.getcwd()
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_PATH, '..', 'data', 'preprocessed', 'analysis_data'))
RESULTS_DIR = os.path.abspath(os.path.join(SCRIPT_PATH, '..', 'results'))
FEATURES_NAMES_DIR = os.path.abspath(os.path.join(DATA_DIR,'feature_names.pkl'))
MODELS_DIR = os.path.abspath(os.path.join(SCRIPT_PATH, '..', 'models'))
os.makedirs(RESULTS_DIR, exist_ok=True)

# === LOAD DATA ===
train_path = os.path.join(DATA_DIR, 'train.csv')
test_path = os.path.join(DATA_DIR, 'test.csv')

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)
target_col = 'Fault_Condition'
feature_cols = joblib.load(FEATURES_NAMES_DIR)

X_train = train_data[feature_cols]
y_train_raw = train_data[target_col]
X_test = test_data[feature_cols]
y_test_raw = test_data[target_col]

le = LabelEncoder()
y_train = le.fit_transform(y_train_raw)
y_test = le.transform(y_test_raw)
print("Mapowanie klas:", dict(zip(le.classes_, le.transform(le.classes_))))
y_map = le.classes_
print(f"Features count: {len(feature_cols)}")
print(f"Training samples count: {len(X_train)}")

results = {'Feature': feature_cols}
# --- Random Forest ---
rf_model = joblib.load(os.path.join(MODELS_DIR, 'random_forest', 'random_forest.pkl'))
results['Random_Forest_Score'] = rf_model.feature_importances_

# --- XGBoost ---
xgb_model = joblib.load(os.path.join(MODELS_DIR, 'XGBoost', 'XGBoost.pkl'))
results['XGBoost_Score'] = xgb_model.feature_importances_

# --- Gaussian Naive Bayes ---
gnb_model = joblib.load(os.path.join(MODELS_DIR, 'naive_bayes', 'gaussian_naive_bayes.pkl'))
perm_gnb = permutation_importance(gnb_model, X_test, y_test, n_repeats=10, random_state=42)
results['GaussianNB_Score'] = perm_gnb.importances_mean

# --- QDA ---
qda_model = joblib.load(os.path.join(MODELS_DIR, 'qda', 'qda_model.pkl'))
perm_qda = permutation_importance(qda_model, X_test, y_test, n_repeats=10, random_state=42)
results['QDA_Score'] = perm_qda.importances_mean

# --- Bayesian ---
ds_post = xr.open_dataset(os.path.join(MODELS_DIR, 'bayesian', 'bayesian_trace.nc'), group='posterior')
W = ds_post['W'].values
results['Bayesian_Score'] = np.mean(np.abs(np.mean(W, axis=(0, 1))), axis=1)
ds_post.close()

df_results = pd.DataFrame(results)


def scale_to_sum_one(series):
    s_clipped = series.clip(lower=0)
    total_sum = s_clipped.sum()
    if total_sum == 0:
        return s_clipped
    return s_clipped / total_sum


df_results['RF_Scaled'] = scale_to_sum_one(df_results['Random_Forest_Score'])
df_results['XGB_Scaled'] = scale_to_sum_one(df_results['XGBoost_Score'])
df_results['GNB_Scaled'] = scale_to_sum_one(df_results['GaussianNB_Score'])
df_results['QDA_Scaled'] = scale_to_sum_one(df_results['QDA_Score'])
df_results['Bayesian_Scaled'] = scale_to_sum_one(df_results['Bayesian_Score'])

df_results['Total_Score'] = (
        df_results['RF_Scaled'] +
        df_results['XGB_Scaled'] +
        df_results['GNB_Scaled'] +
        df_results['QDA_Scaled'] +
        df_results['Bayesian_Scaled']
) / 5.0

df_final_ranking = df_results.sort_values(by='Total_Score', ascending=False).reset_index(drop=True)
cols_order = ['Feature', 'Total_Score',
              'RF_Scaled', 'XGB_Scaled', 'GNB_Scaled', 'QDA_Scaled', 'Bayesian_Scaled',
              'Random_Forest_Score', 'XGBoost_Score', 'GaussianNB_Score', 'QDA_Score', 'Bayesian_Score']

df_final_ranking = df_final_ranking[cols_order]

print("\n--- Features importances rank ---\n")
print(df_final_ranking[['Feature', 'Total_Score']].head(10))

output_file = os.path.join(RESULTS_DIR, 'Feature_Importance_Ranking.csv')
df_final_ranking.to_csv(output_file, index=False)