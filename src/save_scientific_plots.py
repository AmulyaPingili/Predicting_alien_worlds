import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
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

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# Save PR Curve
fig, ax = plt.subplots(figsize=(10, 7))
for name, model in models.items():
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler()),
        ('smote', SMOTE(random_state=42)),
        ('model', model)
    ])
    pipeline.fit(X_train, y_train)
    PrecisionRecallDisplay.from_estimator(pipeline, X_test, y_test, ax=ax, name=name)

plt.title("Precision-Recall Comparison (Scientific Validation)")
pr_curve_path = "reports/plots/pr_curve.png"
os.makedirs("reports/plots", exist_ok=True)
plt.savefig(pr_curve_path)
plt.close()

# Save XGBoost Feature Importance
best_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler()),
    ('smote', SMOTE(random_state=42)),
    ('model', best_model)
])
pipeline.fit(X_train, y_train)
importances = pipeline.named_steps['model'].feature_importances_
feature_imp = pd.Series(importances, index=features).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_imp.values, y=feature_imp.index, hue=feature_imp.index, palette="viridis", legend=False)
plt.title("XGBoost Physics-First Feature Importance")
plt.xlabel("Importance Score")
plt.tight_layout()
importance_path = "reports/plots/feature_importance_xgb.png"
plt.savefig(importance_path)
plt.close()

print(f"Validation plots saved to {pr_curve_path} and {importance_path}")
