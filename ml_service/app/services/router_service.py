"""
Intelligent Router Service for Patient Triage
"""
from typing import Dict, Optional
from app.services.llm_service import get_llm_service
from app.services.supabase_client import get_supabase_service
from app.services.trend_service import get_trend_service

class RouterService:
    """Routes patients between ML Risk 1Engine and LLM Symptom Analyzer"""
    
    def __init__(self):
        self.llm_service = get_llm_service()
        self.supabase = get_supabase_service()
        self.trend_service = get_trend_service()
        
    async def route_patient(
        self,
        patient_id: str,
        risk_score: float,
        risk_level: str,
        symptoms_text: str
    ) -> Dict:
        """
        Decide routing based on risk score and symptoms
        
        Args:
            patient_id: Patient UUID
            risk_score: Diabetes risk score (0-100)
            risk_level: Risk category
            symptoms_text: Description of symptoms
            
        Returns:
            Routing decision and analysis
        """
        # 1. Decision Logic
        # Rule: IF risk_score > 15% OR risk_level is NOT 'Low' -> Dashboard
        is_high_risk = risk_score > 15.0 or risk_level != "Low"
        
        routing_decision = {
            "action": "dashboard" if is_high_risk else "analyzer",
            "risk_score": risk_score,
            "risk_level": risk_level,
            "analysis": None,
            "is_safe": True
        }
        
        llm_analysis = None
        
        # 2. Execute Routing
        if is_high_risk:
            # Route to Dashboard explicitly
            print(f"🚦 Routing High Risk Patient ({risk_score}%) to Dashboard")
            routing_decision["message"] = "High diabetes risk detected. Refer to Clinical Dashboard."
        
        else:
            # Low Risk -> Route to LLM Analyzer
            print(f"🚦 Routing Low Risk Patient ({risk_score}%) to Symptom Analyzer")
            
            # Fetch historical context for LLM
            trend_data = await self.trend_service.get_risk_trajectory(patient_id)
            history_context = self._format_history_for_llm(trend_data)
            
            # Call Gemini
            analysis_result = self.llm_service.analyze_symptoms(
                symptoms_text, 
                patient_history=history_context
            )
            
            routing_decision["analysis"] = analysis_result["analysis"]
            routing_decision["is_safe"] = analysis_result["is_safe"]
            llm_analysis = analysis_result["analysis"]

        # 3. Log Decision
        await self.supabase.log_symptom_analysis(
            patient_id=patient_id,
            symptoms_text=symptoms_text,
            risk_score=risk_score,
            is_diabetic_risk_low=not is_high_risk,
            llm_analysis=llm_analysis,
            routed_to=routing_decision["action"].title()
        )
        
        return routing_decision

    def _format_history_for_llm(self, trend_data: Dict) -> str:
        """Format trend data for LLM prompt"""
        if not trend_data.get('history'):
            return "No previous risk history available."
            
        trend_desc = f"Risk Trend: {trend_data['trend'].upper()} (Delta: {trend_data['delta']})."
        history_points = ", ".join([
            f"{h['date']}: {h['score']}%" 
            for h in trend_data['history'][-3:] # Last 3 points
        ])
        return f"{trend_desc} Recent scores: {history_points}"

    async def get_chat_context(self, patient_id: str) -> Dict:
        """Fetch chat context for patient"""
        trend_data = await self.trend_service.get_risk_trajectory(patient_id)
        
        return {
            "risk_score": trend_data.get("current_score"),
            "trend": trend_data.get("trend"),
            "top_factors": trend_data.get("top_factors", [])
        }


# Global instance
_router_service = None

def get_router_service():
    global _router_service
    if _router_service is None:
        _router_service = RouterService()
    return _router_service
