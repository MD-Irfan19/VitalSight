"""
LLM Service for Symptom Analysis using Google Gemini
"""
import os
import google.generativeai as genai
from typing import Dict, Optional, List
from dotenv import load_dotenv

load_dotenv()

class LLMService:
    """Service for interacting with Google Gemini API"""
    
    def __init__(self):
        """Initialize Gemini client"""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("⚠️  GEMINI_API_KEY not found in environment variables")
            self.model = None
        else:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            print("✅ Gemini LLM service initialized")
            
    def analyze_symptoms(self, symptoms_text: str, patient_history: str = "") -> Dict[str, str]:
        """
        Analyze symptoms using Gemini
        
        Args:
            symptoms_text: The user's described symptoms
            patient_history: Optional context strings
            
        Returns:
            Dict containing 'analysis' and 'is_safe' flag
        """
        if not self.model:
            return {
                "analysis": "LLM service not configured.",
                "is_safe": True
            }

        # Safety Guardrails
        unsafe_keywords = ["chest pain", "difficulty breathing", "unconscious", "stroke", "heart attack"]
        if any(keyword in symptoms_text.lower() for keyword in unsafe_keywords):
            return {
                "analysis": "⚠️ EMERGENCY WARNING: The symptoms described (e.g., chest pain, difficulty breathing) may indicate a life-threatening emergency. Please call emergency services (911 or local equivalent) immediately. This AI system cannot diagnose or treat medical emergencies.",
                "is_safe": False
            }

        prompt = f"""
        You are a clinical decision support assistant for the VitalSight system.
        
        Context:
        - The patient has been assessed as LOW RISK for diabetes by our ML engine.
        - You are now analyzing their symptoms to provide general health insights and educational information.
        
        Strict Instructions: 
        1. Categorize the symptoms (e.g., Respiratory, Gastrointestinal, Musculoskeletal, etc.).
        2. Provide grounded educational insights about these symptoms.
        3. DO NOT provide a medical diagnosis.
        4. DO NOT suggest specific medications.
        5. Maintain a professional, empathetic, and clinical tone.
        
        Patient Symptoms: "{symptoms_text}"
        
        Patient Trend Context: {patient_history}
        
        Response:
        """
        
        try:
            response = self.model.generate_content(prompt)
            return {
                "analysis": response.text,
                "is_safe": True
            }
        except Exception as e:
            print(f"❌ Gemini Generation Error: {e}")
            return {
                "analysis": "I apologize, but I am unable to analyze these symptoms at the moment. Please consult a healthcare professional.",
                "is_safe": True
            }

    def chat_with_context(self, message: str, context: Dict) -> str:
        """
        Chat with context-aware LLM
        
        Args:
            message: User question
            context: Dictionary containing risk_score, trends, shap_values
            
        Returns:
            LLM response string
        """
        if not self.model:
            return "Chat service unavailable."
            
        # Construct context string
        context_str = f"""
        Patient Risk Profile:
        - Current Risk Score: {context.get('risk_score', 'N/A')}%
        - Risk Trend: {context.get('trend', 'Unknown')}
        - Top Risk Factors: {context.get('top_factors', 'N/A')}
        """
        
        prompt = f"""
        You are VitalSight, an AI health assistant.
        
        {context_str}
        
        User Question: "{message}"
        
        Instructions:
        1. Answer the question using the patient's specific risk data.
        2. If asking "Why?", use the Top Risk Factors to explain.
        3. Keep answers concise and helpful.
        4. DO NOT diagnose.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"❌ Chat Error: {e}")
            return "I'm sorry, I cannot answer that right now."


# Global instance
_llm_service = None

def get_llm_service():
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
