import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
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

# Pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler()),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Fit and Predict
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plot_path = "reports/plots/confusion_matrix.png"
os.makedirs("reports/plots", exist_ok=True)
plt.savefig(plot_path)
plt.close()

# Plot Feature Importance
importances = pipeline.named_steps['classifier'].feature_importances_
feature_imp = pd.Series(importances, index=features).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_imp.values, y=feature_imp.index, hue=feature_imp.index, palette="viridis", legend=False)
plt.title("Feature Importance (Physics-First Model)")
plt.xlabel("Importance Score")
plt.tight_layout()
importance_path = "reports/plots/feature_importance.png"
plt.savefig(importance_path)
plt.close()

print(f"Validation plots saved to {plot_path} and {importance_path}")
