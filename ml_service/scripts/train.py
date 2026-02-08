"""
Model training script for diabetes risk prediction
Optimized for high recall to minimize false negatives
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from scripts.preprocess import load_dataset, preprocess_features
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import joblib



def train_xgboost_model(X_train, y_train, X_test, y_test):
    """Train XGBoost model optimized for high recall"""
    # Calculate scale_pos_weight for imbalanced classes
    neg_count = sum(y_train == 0)
    pos_count = sum(y_train == 1)
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
    
    # XGBoost parameters optimized for recall
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'scale_pos_weight': scale_pos_weight,
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    return model


def train_catboost_model(X_train, y_train, X_test, y_test):
    """Train CatBoost model optimized for high recall"""
    # Calculate class weights
    neg_count = sum(y_train == 0)
    pos_count = sum(y_train == 1)
    class_weight = {0: 1.0, 1: neg_count / pos_count if pos_count > 0 else 1.0}
    
    model = cb.CatBoostClassifier(
        iterations=200,
        depth=6,
        learning_rate=0.1,
        loss_function='Logloss',
        class_weights=class_weight,
        random_seed=42,
        verbose=50
    )
    
    model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=50)
    
    return model


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluate model performance with focus on recall"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"\n{'='*60}")
    print(f"{model_name} Evaluation")
    print(f"{'='*60}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(f"True Negatives: {cm[0][0]}")
    print(f"False Positives: {cm[0][1]}")
    print(f"False Negatives: {cm[1][0]} ⚠️  (CRITICAL - we want to minimize this)")
    print(f"True Positives: {cm[1][1]}")
    
    # Calculate metrics
    recall = cm[1][1] / (cm[1][1] + cm[1][0]) if (cm[1][1] + cm[1][0]) > 0 else 0
    precision = cm[1][1] / (cm[1][1] + cm[0][1]) if (cm[1][1] + cm[0][1]) > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"📊 Key Metrics:")
    print(f"{'='*60}")
    print(f"Recall (Sensitivity): {recall:.2%} 🎯")
    print(f"Precision: {precision:.2%}")
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    return recall, precision


def main():
    """Main training pipeline"""
    print("🚀 Starting Diabetes Risk Prediction Model Training")
    print("="*60)
    
    # Paths
    project_root = Path(__file__).parent.parent
    dataset_path = project_root.parent / "diabetes_risk_prediction_dataset.csv"
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Load and preprocess data
    print("\n📊 Loading dataset...")
    df = load_dataset(str(dataset_path))
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution:\n{df['class'].value_counts()}")
    
    print("\n🔧 Preprocessing features...")
    X, y, feature_names = preprocess_features(df)
    print(f"Features: {len(feature_names)}")
    print(f"Feature names: {feature_names}")
    
    # Split data
    print("\n✂️  Splitting data (80/20 train/test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Train XGBoost
    print("\n🌲 Training XGBoost model...")
    model = train_xgboost_model(X_train, y_train, X_test, y_test)
    recall, precision = evaluate_model(model, X_test, y_test, "XGBoost")
    
    print(f"\n✅ Model trained: XGBoost (Recall: {recall:.2%})")

    
    # Save the model
    model_path = models_dir / "diabetes_model.pkl"
    print(f"\n💾 Saving model to {model_path}...")
    joblib.dump({
        'model': model,
        'feature_names': feature_names,
        'model_type': "XGBoost",
        'recall': recall
    }, model_path)

    
    print("\n✅ Training complete!")
    print(f"Model saved at: {model_path}")
    print("\n" + "="*60)
    print("🎯 Model is optimized for HIGH RECALL to minimize false negatives")
    print("="*60)


if __name__ == "__main__":
    main()
