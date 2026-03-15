import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import PrecisionRecallDisplay
import os

# Set theme
sns.set_theme(style="whitegrid")

# Load data
data_path = "data/processed_exoplanet_data.csv"
df = pd.read_csv(data_path)

# Feature selection
features = ['pl_orbper', 'pl_orbsmax', 'pl_rade', 'pl_insol', 
            'st_teff', 'st_logg', 'st_rad', 'sy_dist', 'sy_vmag']
target = 'is_habitable'

X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

def get_pipeline(model):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler()),
        ('smote', SMOTE(random_state=42)),
        ('model', model)
    ])

models_dict = {
    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42),
    'CatBoost': CatBoostClassifier(verbose=0, random_state=42),
    'LightGBM': LGBMClassifier(random_state=42, verbose=-1),
    'Stacked Ensemble': StackingClassifier(
        estimators=[
            ('xgb', XGBClassifier(eval_metric='logloss', random_state=42)),
            ('cat', CatBoostClassifier(verbose=0, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
        ],
        final_estimator=LogisticRegression(),
        cv=5
    )
}

# Save Advanced PR Curve
fig, ax = plt.subplots(figsize=(12, 8))
for name, model in models_dict.items():
    pipe = get_pipeline(model)
    pipe.fit(X_train, y_train)
    PrecisionRecallDisplay.from_estimator(pipe, X_test, y_test, ax=ax, name=name)

plt.title("Advanced Benchmarking: Precision-Recall Comparison")
pr_curve_path = "reports/plots/advanced_pr_curve.png"
os.makedirs("reports/plots", exist_ok=True)
plt.savefig(pr_curve_path)
plt.close()

print(f"Advanced validation plot saved to {pr_curve_path}")
