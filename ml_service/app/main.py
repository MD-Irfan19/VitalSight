"""
FastAPI application for diabetes risk prediction
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

from app.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
    TopFactor,
    HealthStatus,
    ChatRequest,
    ChatResponse
)
from app.models.predictor import get_predictor
from app.services.explainability import get_explainer, initialize_explainer
from app.services.supabase_client import get_supabase_service
from app.services.router_service import get_router_service

from scripts.preprocess import encode_symptoms_dict, load_dataset, preprocess_features
import numpy as np


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup"""
    print("🚀 Starting VitalSight ML Service...")
    
    # Load model
    predictor = get_predictor()
    if not predictor.is_loaded():
        print("❌ Model not found. Please train the model first:")
        print("   python scripts/train.py")
    else:
        print(f"✅ Model loaded: {predictor.model_type}")
        
        # Initialize explainer with training data
        try:
            dataset_path = Path(__file__).parent.parent.parent / "diabetes_risk_prediction_dataset.csv"
            if dataset_path.exists():
                df = load_dataset(str(dataset_path))
                X_train, _, _ = preprocess_features(df)
                initialize_explainer(
                    predictor.model,
                    predictor.feature_names,
                    X_train
                )
        except Exception as e:
            print(f"⚠️  Could not initialize explainer with training data: {e}")
            initialize_explainer(predictor.model, predictor.feature_names)
    
    # Initialize Supabase
    supabase = get_supabase_service()
    
    print("✅ VitalSight ML Service ready!")
    
    yield
    
    # Shutdown
    print("👋 Shutting down VitalSight ML Service...")


# Create FastAPI app
app = FastAPI(
    title="VitalSight ML Service",
    description="Diabetes risk prediction with SHAP explainability",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "VitalSight ML Service",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health", response_model=HealthStatus)
async def health_check():
    """Health check endpoint"""
    predictor = get_predictor()
    return HealthStatus(
        status="healthy",
        model_loaded=predictor.is_loaded(),
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_diabetes_risk(request: PredictionRequest):
    """
    Predict diabetes risk for a patient
    
    Args:
        request: Prediction request with patient_id and symptoms
        
    Returns:
        Prediction response with risk score, confidence, and explanations
    """
    try:

        # Get services
        predictor = get_predictor()
        if not predictor.is_loaded():
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please train the model first."
            )
        
        explainer = get_explainer()
        supabase = get_supabase_service()
        router = get_router_service()
        
        # Convert symptoms to dict for encoding
        symptoms_dict = request.symptoms.dict()
        
        # Encode features
        features = encode_symptoms_dict(symptoms_dict)
        
        # Get prediction
        risk_score, risk_level = predictor.predict(features)
        
        # Get SHAP explanation
        top_factors_data = explainer.get_explanation(features, symptoms_dict)
        top_factors = [TopFactor(**factor) for factor in top_factors_data]
        
        # Calculate confidence score
        confidence_score = explainer.calculate_confidence_score(features)
        
        # Store health vitals (optional, may fail if tables don't exist)
        vitals_id = None
        if supabase.is_connected():
            print("📝 Storing health vitals...")
            vitals_id = await supabase.store_health_vitals(
                request.patient_id,
                symptoms_dict
            )
            # if vitals_id:
            #     print(f"✅ Vitals stored: {vitals_id}")
            # else:
            #     print("⚠️  Failed to store vitals (check logs)")
        
        # Store prediction
        supabase_result = None
        if supabase.is_connected():
            print("📝 storing prediction...")
            supabase_result = await supabase.write_prediction(
                patient_id=request.patient_id,
                risk_score=risk_score,
                risk_level=risk_level,
                confidence_score=confidence_score,
                shap_values=top_factors_data,
                vitals_id=vitals_id
            )
            if supabase_result.get('success'):
                 print(f"✅ Prediction stored: {supabase_result.get('prediction_id')}")
            else:
                 print(f"❌ Failed to store prediction: {supabase_result.get('error')}")

        # Intelligent Routing
        symptoms_text = request.symptoms_text
        if not symptoms_text:
            # Generate default description from structured symptoms
            positive_symptoms = [
                k.replace('_', ' ').title() 
                for k, v in symptoms_dict.items() 
                if v == "Yes" and k not in ['age', 'gender']
            ]
            symptoms_text = f"Patient is a {symptoms_dict['age']} year old {symptoms_dict['gender']} reporting: {', '.join(positive_symptoms)}."
            
        print(f"🧭 Routing patient based on Risk Score: {risk_score:.1f}%")
        routing_decision = await router.route_patient(
            patient_id=request.patient_id,
            risk_score=risk_score,
            risk_level=risk_level,
            symptoms_text=symptoms_text
        )

        
        # Create response
        from app.schemas.prediction import RoutingInfo
        
        response = PredictionResponse(
            patient_id=request.patient_id,
            risk_score=round(risk_score, 2),
            risk_level=risk_level,
            confidence_score=confidence_score,
            top_factors=top_factors,
            routing=RoutingInfo(**routing_decision)
        )

        
        # Log prediction
        print(f"\n{'='*60}")
        print(f"✅ Prediction Complete")
        print(f"{'='*60}")
        print(f"Patient ID: {request.patient_id}")
        print(f"Risk Score: {risk_score:.1f}%")
        print(f"Risk Level: {risk_level}")
        print(f"Confidence: {confidence_score}")
        print(f"Top Factors:")
        for i, factor in enumerate(top_factors, 1):
            print(f"  {i}. {factor.feature}: {factor.value} (Impact: {factor.impact:+.3f})")
        print(f"{'='*60}\n")
        
        return response
        
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


from app.services.llm_service import get_llm_service

# ... (inside imports or top level)

@app.post("/chat", response_model=ChatResponse)
async def chat_with_bot(request: ChatRequest):
    """
    Context-aware chat endpoint
    
    Args:
        request: Chat request with patient_id and message
        
    Returns:
        LLM response
    """
    try:
        router = get_router_service()
        llm = get_llm_service()
        
        # Get patient context (risk score, trends, etc)
        context = await router.get_chat_context(request.patient_id)
        
        # Get LLM response
        response_text = llm.chat_with_context(request.message, context)
        
        return ChatResponse(
            response=response_text,
            context_used=True if context.get('risk_score') is not None else False
        )
        
    except Exception as e:
        print(f"❌ Chat error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Chat failed: {str(e)}"
        )


if __name__ == "__main__":

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
