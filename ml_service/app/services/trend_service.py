"""
Trend Service for analyzing patient risk trajectory
"""
from typing import List, Dict, Optional
from datetime import datetime
from app.services.supabase_client import get_supabase_service

class TrendService:
    """Service to calculate risk score trends"""
    
    def __init__(self):
        self.supabase = get_supabase_service()
        
    async def get_risk_trajectory(self, patient_id: str) -> Dict:
        """
        Get risk trajectory for a patient
        
        Args:
            patient_id: Patient UUID
            
        Returns:
            Dict with current score, previous score, delta, and history
        """
        history = await self.supabase.get_patient_history(patient_id, limit=5)
        
        if not history:
            return {
                "current_score": None,
                "previous_score": None,
                "delta": 0,
                "trend": "stable",
                "history": []
            }
            
        current = history[0]
        previous = history[1] if len(history) > 1 else None
        
        current_score = float(current.get('risk_score', 0))
        previous_score = float(previous.get('risk_score', 0)) if previous else None
        
        delta = 0
        trend = "stable"
        
        if previous_score is not None:
            delta = current_score - previous_score
            if delta > 5:
                trend = "deteriorating"
            elif delta < -5:
                trend = "improving"
                
        # Format history for frontend chart
        # [ { "date": "2023-10-27", "score": 45 }, ... ]
        formatted_history = []
        for record in reversed(history): # Chronological order for charts
            date_str = record.get('created_at', '')[:10] # YYYY-MM-DD
            formatted_history.append({
                "date": date_str,
                "score": float(record.get('risk_score', 0))
            })
            
        return {
            "current_score": current_score,
            "previous_score": previous_score,
            "delta": round(delta, 2),
            "trend": trend,
            "top_factors": history[0].get('shap_values', []),
            "history": formatted_history
        }

# Global instance
_trend_service = None

def get_trend_service():
    global _trend_service
    if _trend_service is None:
        _trend_service = TrendService()
    return _trend_service
