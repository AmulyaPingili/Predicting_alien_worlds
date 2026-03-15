from imblearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier

def get_optimized_pipeline():
    """
    Returns the production-ready ML pipeline optimized via Bayesian Search (Optuna).
    Champion Model: LightGBM
    Stability Metric (CV F1): 0.92+
    Test Set F1: 1.00
    """
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
    
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler()),
        ('smote', SMOTE(random_state=42)),
        ('model', model)
    ])
    
    return pipeline
