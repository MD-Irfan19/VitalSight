"""
Supabase client for storing predictions
"""
import os
from supabase import create_client, Client
from typing import Dict, Optional, List
from dotenv import load_dotenv
import json
from datetime import datetime

# Load environment variables
load_dotenv()


class SupabaseService:
    """Supabase integration for storing predictions"""
    
    def __init__(self):
        """Initialize Supabase client"""
        self.url = os.getenv('SUPABASE_URL')
        self.key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
        
        if not self.url or not self.key:
            print("⚠️  Warning: Supabase credentials not configured")
            self.client = None
        else:
            self.client: Client = create_client(self.url, self.key)
            print("✅ Supabase client initialized")
    
    def is_connected(self) -> bool:
        """Check if Supabase is connected"""
        return self.client is not None
    
    async def write_prediction(
        self,
        patient_id: str,
        risk_score: float,
        risk_level: str,
        confidence_score: str,
        shap_values: list,
        vitals_id: Optional[str] = None
    ) -> Dict:
        """
        Write prediction to Supabase risk_predictions table
        
        Args:
            patient_id: Patient UUID
            risk_score: Risk probability percentage
            risk_level: Risk category (Low/Medium/High)
            confidence_score: Prediction confidence (High/Medium/Low)
            shap_values: Top contributing factors from SHAP
            vitals_id: Optional reference to health_vitals record
            
        Returns:
            Inserted record or error dict
        """
        if not self.is_connected():
            return {
                'error': 'Supabase not configured',
                'success': False
            }
        
        try:
            # Map confidence string to numeric if needed, or just try to insert
            # The existing schema has confidence_score as double precision
            # We'll need to adapt. Let's assume High=1.0, Medium=0.5, Low=0.0 for now
            # or better, just use 1.0/0.0 if we can't change the schema.
            # actually, let's try to map it.
            conf_map = {"High": 0.95, "Medium": 0.5, "Low": 0.1}
            conf_numeric = conf_map.get(confidence_score, 0.5)

            # Prepare data matching the EXISTING schema
            data = {
                'patient_id': patient_id,
                'risk_score': round(risk_score, 2),
                'risk_category': risk_level,  # Column name is risk_category in DB
                'confidence_score': conf_numeric, # Column is double precision in DB
                'shap_values': shap_values,
                'created_at': datetime.utcnow().isoformat()
            }
            
            # Note: vitals_id and model_version seem missing from the existing table schema
            # based on my check. I'll omit them for now to ensure writing works.
            
            # Insert into risk_predictions table
            result = self.client.table('risk_predictions').insert(data).execute()

            
            print(f"✅ Prediction saved to Supabase (ID: {result.data[0]['id']})")
            
            return {
                'success': True,
                'prediction_id': result.data[0]['id'],
                'data': result.data[0]
            }
            
        except Exception as e:
            print(f"❌ Error writing to Supabase: {str(e)}")
            return {
                'error': str(e),
                'success': False
            }
    
    async def store_health_vitals(
        self,
        patient_id: str,
        symptoms: Dict
    ) -> Optional[str]:
        """
        Store health vitals in database
        
        Args:
            patient_id: Patient UUID
            symptoms: Symptom dictionary
            
        Returns:
            Vitals record ID or None
        """
        if not self.is_connected():
            return None
        
        try:
            # SKIP writing to health_vitals because the existing table schema is for
            # numerical vitals (glucose, bmi, etc) and does not match our symptom data.
            # To avoid errors, we skip this step.
            print("⚠️  Skipping health_vitals storage: Schema mismatch (Table expects numerical vitals)")
            return None

            # Original code disabled:
            # Prepare vitals data
            # data = {
            #     'patient_id': patient_id,
            #     'age': symptoms.get('age'),
            #     'gender': symptoms.get('gender'),
            #     'polyuria': symptoms.get('polyuria'),
            #     ...
            # }
            # result = self.client.table('health_vitals').insert(data).execute()
            # return result.data[0]['id']

            
            # return result.data[0]['id']
            
            vitals_id = result.data[0]['id']
            print(f"✅ Health vitals saved (ID: {vitals_id})")
            
            return vitals_id
            
        except Exception as e:
            print(f"⚠️  Could not store health vitals: {str(e)}")
            return None



    async def get_patient_history(self, patient_id: str, limit: int = 5) -> List[Dict]:
        """
        Fetch recent risk predictions for a patient
        
        Args:
            patient_id: Patient UUID
            limit: Number of records to fetch
            
        Returns:
            List of prediction records ordered by date
        """
        if not self.is_connected():
            return []
            
        try:
            # Fetch from risk_predictions table
            response = self.client.table('risk_predictions') \
                .select('risk_score, created_at, risk_category, shap_values') \
                .eq('patient_id', patient_id) \
                .order('created_at', desc=True) \
                .limit(limit) \
                .execute()
                
            return response.data if response.data else []
            
        except Exception as e:
            print(f"❌ Error fetching history: {e}")
            return []


    async def log_symptom_analysis(
        self,
        patient_id: Optional[str],
        symptoms_text: str,
        risk_score: float,
        is_diabetic_risk_low: bool,
        llm_analysis: Optional[str],
        routed_to: str
    ) -> bool:
        """
        Log symptom analysis and routing decision
        """
        if not self.is_connected():
            return False
            
        try:
            data = {
                "patient_id": patient_id,
                "symptoms_text": symptoms_text,
                "risk_score": round(risk_score, 2),
                "is_diabetic_risk_low": is_diabetic_risk_low,
                "llm_insight": llm_analysis,
                "routed_to": routed_to,
                "created_at": datetime.utcnow().isoformat()
            }
            
            self.client.table('symptom_logs').insert(data).execute()
            print("✅ Symptom analysis logged")
            return True
            
        except Exception as e:
            print(f"❌ Error logging symptom analysis: {e}")
            return False

# Global Supabase service instance
_supabase_service = None



def get_supabase_service() -> SupabaseService:
    """Get or create global Supabase service instance"""
    global _supabase_service
    if _supabase_service is None:
        _supabase_service = SupabaseService()
    return _supabase_service

