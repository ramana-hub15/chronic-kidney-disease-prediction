import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
import joblib
import json
import os

def clean_dm(val):
    if pd.isna(val):
        return val
    val = str(val).strip().lower()
    if val in ['yes', ' y', 'yes\t']:
        return 1
    elif val in ['no', '\tno']:
        return 0
    return np.nan

def clean_htn(val):
    if pd.isna(val):
        return val
    val = str(val).strip().lower()
    if val == 'yes':
        return 1
    elif val == 'no':
        return 0
    return np.nan

def clean_classification(val):
    if pd.isna(val):
        return val
    val = str(val).strip().lower()
    if val.startswith('ckd'):
        return 1
    elif val == 'notckd':
        return 0
    return np.nan

def clean_pcv(val):
    if pd.isna(val):
        return np.nan
    val = str(val).strip()
    if val == '' or val == '\t?':
        return np.nan
    try:
        return float(val)
    except:
        return np.nan

def main():
    print("Loading data...")
    df = pd.read_csv('kidney_disease.csv')
    
    # 9 features requested + classification
    features = ['sc', 'bu', 'hemo', 'pcv', 'sg', 'al', 'bgr', 'htn', 'dm']
    target = 'classification'
    
    # Keep only required columns
    df = df[features + [target]].copy()
    
    print("Cleaning data...")
    df['dm'] = df['dm'].apply(clean_dm)
    df['htn'] = df['htn'].apply(clean_htn)
    df['classification'] = df['classification'].apply(clean_classification)
    df['pcv'] = df['pcv'].apply(clean_pcv)
    
    # Convert numerical columns to float explicitly
    num_cols = ['sc', 'bu', 'hemo', 'pcv', 'sg', 'al', 'bgr']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Impute missing values
    # For numerical columns: median
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
        
    # For categorical columns: mode
    for col in ['htn', 'dm']:
        df[col] = df[col].fillna(df[col].mode()[0])
        
    df = df.dropna(subset=['classification'])
    
    X = df[features]
    y = df[target]
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Scaling data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training ANN with calibrated probabilities...")
    # Use a smaller network with regularization to prevent overfitting
    base_ann = MLPClassifier(
        hidden_layer_sizes=(32, 16), 
        max_iter=500, 
        random_state=42,
        alpha=0.01,          # L2 regularization to prevent overfitting
        early_stopping=True, # Stop when validation score doesn't improve
        validation_fraction=0.15,
        n_iter_no_change=20
    )
    # Wrap with CalibratedClassifierCV for well-calibrated probabilities
    ann = CalibratedClassifierCV(base_ann, cv=5, method='sigmoid')
    ann.fit(X_train_scaled, y_train)
    
    print("Training Logistic Regression with calibrated probabilities...")
    base_lr = LogisticRegression(random_state=42, C=0.5)  # Add regularization
    lr = CalibratedClassifierCV(base_lr, cv=5, method='sigmoid')
    lr.fit(X_train_scaled, y_train)
    
    print("Evaluating models...")
    models = {'ANN': ann, 'Logistic Regression': lr}
    metrics = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        metrics[name] = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred)),
            'recall': float(recall_score(y_test, y_pred)),
            'f1_score': float(f1_score(y_test, y_pred)),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        print(f"  {name}: Accuracy={metrics[name]['accuracy']:.4f}, "
              f"Prob range=[{y_prob.min():.4f}, {y_prob.max():.4f}]")
    
    os.makedirs('models', exist_ok=True)
    print("Saving models and metrics...")
    joblib.dump(ann, 'models/ann_model.pkl')
    joblib.dump(lr, 'models/lr_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    with open('models/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print("Training complete! Models saved in 'models/' directory.")

if __name__ == '__main__':
    main()

