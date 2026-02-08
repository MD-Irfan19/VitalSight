"""
Debug script for Supabase connection
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client, Client
from datetime import datetime
import uuid

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def test_connection():
    # 1. Load environment variables
    env_path = Path(__file__).parent.parent / '.env'
    print(f"Loading .env from: {env_path}")
    
    if not env_path.exists():
        print("❌ .env file not found!")
        return

    load_dotenv(env_path)
    
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    
    print(f"URL: {url}")
    print(f"Key (first 10 chars): {key[:10] if key else 'None'}")
    
    if not url or not key:
        print("❌ Missing Supabase credentials in .env")
        return

    # 2. Connect to Supabase
    try:
        supabase: Client = create_client(url, key)
        print("✅ Supabase client created")
    except Exception as e:
        print(f"❌ Failed to create client: {e}")
        return

    # 3. Test Write (Risk Prediction)
    try:
        # Use a valid patient_id that exists in the 'patients' table
        test_patient_id = "123e4567-e89b-12d3-a456-426614174000"
        print(f"Testing write with patient_id: {test_patient_id}")

        
        data = {
            'patient_id': test_patient_id,
            'risk_score': 50.0,
            'risk_category': 'Medium',  # Schema mismatch fix
            'confidence_score': 0.8,    # Schema mismatch fix (numeric)
            'shap_values': [{'feature': 'Test', 'value': 'Yes', 'impact': 0.5}],
            'created_at': datetime.utcnow().isoformat()
        }

        
        result = supabase.table('risk_predictions').insert(data).execute()
        print(f"✅ Implementation SUCCESS! Written ID: {result.data[0]['id']}")
        
    except Exception as e:
        print(f"❌ Write failed: {str(e)}")
        # Check if it's an RLS issue or connectivity
        if "policy" in str(e).lower():
            print("👉 Check RLS policies on 'risk_predictions' table")
        elif "connection" in str(e).lower():
            print("👉 Check internet connection or Supabase URL")

if __name__ == "__main__":
    test_connection()
