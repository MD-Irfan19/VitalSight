"""
SHAP-based explainability service for diabetes predictions
"""
import shap
import numpy as np
import pandas as pd
from typing import List, Dict
from sklearn.ensemble import IsolationForest


class ExplainabilityService:
    """SHAP explainability and confidence scoring"""
    
    def __init__(self, model, feature_names: List[str], training_data: np.ndarray = None):
        """
        Initialize explainability service
        
        Args:
            model: Trained model
            feature_names: List of feature names
            training_data: Training data for reference distribution (optional)
        """
        self.model = model
        self.feature_names = feature_names
        self.training_data = training_data
        
        # Initialize SHAP explainer
        self.explainer = shap.TreeExplainer(model)
        
        # Initialize outlier detector if training data provided
        self.outlier_detector = None
        if training_data is not None:
            self.outlier_detector = IsolationForest(
                contamination=0.1,
                random_state=42
            )
            self.outlier_detector.fit(training_data)
    
    def get_explanation(self, features: np.ndarray, original_values: Dict) -> List[Dict]:
        """
        Get SHAP explanation for prediction
        
        Args:
            features: Encoded feature array
            original_values: Original symptom values (for display)
            
        Returns:
            List of top 3 contributing factors with SHAP values
        """
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(features)
        
        # Get SHAP values for positive class (diabetes risk)
        if isinstance(shap_values, list):
            # For binary classification, take values for class 1
            shap_vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        else:
            shap_vals = shap_values[0]
        
        # Create feature-value-impact tuples
        explanations = []
        for i, (feature_name, shap_val) in enumerate(zip(self.feature_names, shap_vals)):
            # Get original value
            feature_key = feature_name.lower().replace(' ', '_')
            original_val = original_values.get(feature_key, str(features[0][i]))
            
            explanations.append({
                'feature': feature_name,
                'value': str(original_val),
                'impact': float(shap_val),
                'abs_impact': abs(float(shap_val))
            })
        
        # Sort by absolute impact (descending) and take top 3
        explanations.sort(key=lambda x: x['abs_impact'], reverse=True)
        top_3 = explanations[:3]
        
        # Remove abs_impact from output
        for exp in top_3:
            del exp['abs_impact']
        
        return top_3
    
    def calculate_confidence_score(self, features: np.ndarray) -> str:
        """
        Calculate confidence score based on outlier detection
        
        Args:
            features: Feature array
            
        Returns:
            Confidence level: "High", "Medium", or "Low"
        """
        if self.outlier_detector is None:
            # No training data available, return default
            return "High"
        
        # Predict outlier score (-1 for outliers, 1 for inliers)
        outlier_score = self.outlier_detector.predict(features)[0]
        
        # Decision function gives anomaly score (lower = more abnormal)
        anomaly_score = self.outlier_detector.decision_function(features)[0]
        
        if outlier_score == -1:  # Outlier
            return "Low"
        elif anomaly_score > 0.1:  # Strong inlier
            return "High"
        else:  # Weak inlier
            return "Medium"


# Global explainer instance
_explainer = None
_training_data = None


def initialize_explainer(model, feature_names: List[str], training_data: np.ndarray = None):
    """Initialize global explainer instance"""
    global _explainer, _training_data
    _training_data = training_data
    _explainer = ExplainabilityService(model, feature_names, training_data)
    print("✅ SHAP Explainer initialized")


def get_explainer() -> ExplainabilityService:
    """Get global explainer instance"""
    if _explainer is None:
        raise ValueError("Explainer not initialized. Call initialize_explainer first.")
    return _explainer
