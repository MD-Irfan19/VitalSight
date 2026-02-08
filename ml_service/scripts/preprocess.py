"""
Data preprocessing utilities for diabetes dataset
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def load_dataset(filepath: str) -> pd.DataFrame:
    """Load the diabetes dataset"""
    df = pd.read_csv(filepath)
    return df


def preprocess_features(df: pd.DataFrame) -> tuple:
    """
    Preprocess the diabetes dataset
    
    Returns:
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
    """
    # Make a copy
    data = df.copy()
    
    # Separate features and target
    X = data.drop('class', axis=1)
    y = data['class']
    
    # Encode categorical features
    # Gender: Male=1, Female=0
    X['Gender'] = X['Gender'].map({'Male': 1, 'Female': 0})
    
    # Yes/No features: Yes=1, No=0
    yes_no_columns = [col for col in X.columns if col != 'Age' and col != 'Gender']
    for col in yes_no_columns:
        X[col] = X[col].map({'Yes': 1, 'No': 0})
    
    # Encode target: Positive=1, Negative=0
    y = y.map({'Positive': 1, 'Negative': 0})
    
    # Get feature names
    feature_names = X.columns.tolist()
    
    return X.values, y.values, feature_names


def encode_symptoms_dict(symptoms: dict) -> np.ndarray:
    """
    Encode a symptoms dictionary for prediction
    
    Args:
        symptoms: Dict with keys matching dataset columns
        
    Returns:
        Encoded feature array
    """
    # Feature order (must match training data)
    feature_order = [
        'Age', 'Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss',
        'weakness', 'Polyphagia', 'Genital thrush', 'visual blurring',
        'Itching', 'Irritability', 'delayed healing', 'partial paresis',
        'muscle stiffness', 'Alopecia', 'Obesity'
    ]
    
    # Create feature array
    features = []
    for feat in feature_order:
        # Handle case-insensitive matching
        feat_lower = feat.lower().replace(' ', '_')
        value = symptoms.get(feat_lower, symptoms.get(feat, 'No'))
        
        # Encode
        if feat == 'Age':
            features.append(int(value))
        elif feat == 'Gender':
            features.append(1 if value == 'Male' else 0)
        else:
            features.append(1 if value == 'Yes' else 0)
    
    return np.array(features).reshape(1, -1)
