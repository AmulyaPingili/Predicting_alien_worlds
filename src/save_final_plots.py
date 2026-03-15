import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.metrics import PrecisionRecallDisplay, f1_score, classification_report
import os

# Load data
data_path = "data/processed_exoplanet_data.csv"
df = pd.read_csv(data_path)
features = ['pl_orbper', 'pl_orbsmax', 'pl_rade', 'pl_insol', 
            'st_teff', 'st_logg', 'st_rad', 'sy_dist', 'sy_vmag']
target = 'is_habitable'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

def get_pipeline(model):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler()),
        ('smote', SMOTE(random_state=42)),
        ('model', model)
    ])

# Best parameters from Optuna
best_params = {
    'num_leaves': 125, 
    'learning_rate': 0.045940680368031346, 
    'n_estimators': 712, 
    'min_child_samples': 68, 
    'reg_alpha': 0.010243126069133085, 
    'reg_lambda': 3.1821736384084893e-06, 
    'scale_pos_weight': 187.9941136622789,
    'random_state': 42,
    'verbose': -1
}

model = LGBMClassifier(**best_params)
pipe = get_pipeline(model)
pipe.fit(X_train, y_train)

# Save Final Plot
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 7))
PrecisionRecallDisplay.from_estimator(pipe, X_test, y_test, ax=ax, name='Optimized LightGBM')
plt.title("Final Model Validation: Precision-Recall (Optimized)")
plt.savefig("reports/plots/final_optimized_pr.png")
plt.close()

# Copy to artifacts
artifact_dir = "C:/Users/saiam/.gemini/antigravity/brain/4c39e10c-1c4d-45fb-a4ec-dfd20d674396"
import shutil
shutil.copy("reports/plots/final_optimized_pr.png", os.path.join(artifact_dir, "final_optimized_pr.png"))

print("Final optimized plots saved.")
