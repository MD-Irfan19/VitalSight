"""
Model predictor module for inference
"""
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import os


class DiabetesPredictor:
    """Diabetes risk prediction model wrapper"""
    
    def __init__(self, model_path: str = None):
        """Initialize predictor with model"""
        if model_path is None:
            # Default path
            model_path = os.getenv('MODEL_PATH', 'models/diabetes_model.pkl')
            model_path = Path(__file__).parent.parent.parent / model_path
        
        self.model_data = None
        self.model = None
        self.feature_names = None
        self.model_type = None
        
        if Path(model_path).exists():
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load the trained model"""
        self.model_data = joblib.load(model_path)
        self.model = self.model_data['model']
        self.feature_names = self.model_data['feature_names']
        self.model_type = self.model_data.get('model_type', 'Unknown')
        print(f"✅ Loaded {self.model_type} model from {model_path}")
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None
    
    def predict(self, features: np.ndarray) -> Tuple[float, str]:
        """
        Predict diabetes risk
        
        Args:
            features: Feature array (must match training feature order)
            
        Returns:
            Tuple of (risk_score, risk_level)
        """
        if not self.is_loaded():
            raise ValueError("Model not loaded")
        
        # Get probability for positive class
        risk_proba = self.model.predict_proba(features)[0, 1]
        risk_score = float(risk_proba * 100)  # Convert to percentage
        
        # Determine risk level
        if risk_score < 30:
            risk_level = "Low"
        elif risk_score < 60:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return risk_score, risk_level


# Global predictor instance
_predictor = None


def get_predictor() -> DiabetesPredictor:
    """Get or create global predictor instance"""
    global _predictor
    if _predictor is None:
        _predictor = DiabetesPredictor()
    return _predictor
