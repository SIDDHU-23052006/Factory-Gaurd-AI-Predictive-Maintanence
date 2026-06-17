import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc, classification_report, average_precision_score, confusion_matrix
import xgboost as xgb
from xgboost import XGBClassifier
import optuna
import shap
import matplotlib.pyplot as plt

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def prepare_data(df):
    # Features to drop
    drop_cols = ['timestamp', 'machine_id', 'failure', 'failure_in_24h']
    
    # Target
    y = df['failure_in_24h']
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    
    # Train-test split (chronological split would be better for time series, but for this synthetic data random split is acceptable or split by machine)
    # Let's do a random split for simplicity, but stratify due to imbalance
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    return X_train, X_test, y_train, y_test

def tune_xgboost(X_train, y_train, n_trials=10):
    print("Starting Optuna hyperparameter tuning for XGBoost...")
    
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    print(f"Using scale_pos_weight: {scale_pos_weight:.2f}")

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'scale_pos_weight': scale_pos_weight, # Handle imbalance
            'objective': 'binary:logistic',
            'eval_metric': 'aucpr',
            'random_state': 42,
            'n_jobs': -1
        }
        
        # We can use cross-validation, but to save time, we will just use a simple split
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)
        
        model = XGBClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        
        preds_proba = model.predict_proba(X_val)[:, 1]
        pr_auc = average_precision_score(y_val, preds_proba)
        
        return pr_auc

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    print("Best trial:")
    print("  Value (PR-AUC): ", study.best_trial.value)
    print("  Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
        
    best_params = study.best_trial.params
    best_params['scale_pos_weight'] = scale_pos_weight
    best_params['objective'] = 'binary:logistic'
    best_params['random_state'] = 42
    best_params['n_jobs'] = -1
    
    return best_params

def train_and_evaluate(X_train, X_test, y_train, y_test, params):
    print("Training final XGBoost model...")
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    # Evaluation
    preds = model.predict(X_test)
    preds_proba = model.predict_proba(X_test)[:, 1]
    
    # PR-AUC
    precision, recall, _ = precision_recall_curve(y_test, preds_proba)
    pr_auc = auc(recall, precision)
    
    print(f"Final Model PR-AUC: {pr_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, preds)
    print(cm)
    
    # Save evaluation plots
    os.makedirs('assets', exist_ok=True)
    
    # 1. Confusion Matrix Plot
    from sklearn.metrics import ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Healthy', 'Fail Risk'])
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
    plt.title("Confusion Matrix", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('assets/confusion_matrix.png', dpi=300)
    plt.close()
    print("Saved Confusion Matrix plot to assets/confusion_matrix.png")
    
    # 2. Precision-Recall Curve Plot
    ap = average_precision_score(y_test, preds_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'Precision-Recall Curve (AP = {ap:.4f})', color='#ff5722', lw=2)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve (PR-AUC)', fontsize=14, fontweight='bold', pad=15)
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('assets/pr_curve.png', dpi=300)
    plt.close()
    print("Saved Precision-Recall curve to assets/pr_curve.png")
    
    # 3. Feature Importance Plot
    importances = model.feature_importances_
    features = X_train.columns.tolist()
    indices = np.argsort(importances)[::-1][:15] # Top 15 features
    
    plt.figure(figsize=(10, 6))
    plt.title("Top 15 Feature Importances (XGBoost)", fontsize=14, fontweight='bold', pad=15)
    plt.barh(range(len(indices)), importances[indices], align="center", color="#3f51b5")
    plt.yticks(range(len(indices)), [features[i] for i in indices], fontsize=10)
    plt.xlabel("Relative Importance", fontsize=12)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('assets/feature_importance.png', dpi=300)
    plt.close()
    print("Saved Feature Importance plot to assets/feature_importance.png")
    
    return model

def explain_model(model, X_train):
    print("Generating SHAP values for explainability...")
    # Use a sample for SHAP to save time
    X_sample = X_train.sample(min(1000, len(X_train)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    os.makedirs('reports', exist_ok=True)
    os.makedirs('assets', exist_ok=True)
    
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.savefig('reports/shap_summary.png', bbox_inches='tight')
    plt.savefig('assets/shap_summary.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("SHAP summary plot saved to reports/shap_summary.png and assets/shap_summary.png")


def main():
    if not os.path.exists('data/processed/features.csv'):
        print("Processed data not found. Please run feature_engineering.py first.")
        return
        
    df = load_data('data/processed/features.csv')
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    print(f"Class distribution - Train: Normal={sum(y_train==0)}, Failure={sum(y_train==1)}")
    
    # In production we might use more trials, using 5 for speed here
    best_params = tune_xgboost(X_train, y_train, n_trials=5)
    
    model = train_and_evaluate(X_train, X_test, y_train, y_test, best_params)
    
    # Explainability
    explain_model(model, X_train)
    
    # Save model and feature names for deployment
    os.makedirs('models', exist_ok=True)
    joblib.dump({
        'model': model,
        'features': X_train.columns.tolist()
    }, 'models/xgboost_production.pkl')
    print("Model pipeline saved to models/xgboost_production.pkl")

if __name__ == "__main__":
    main()
