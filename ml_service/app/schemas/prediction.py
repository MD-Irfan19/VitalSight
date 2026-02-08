"""
Pydantic schemas for request/response validation
"""
from pydantic import BaseModel, Field, validator
from typing import Literal, List, Dict
from datetime import datetime
import uuid


class Symptoms(BaseModel):
    """Patient symptom data"""
    age: int = Field(..., ge=0, le=120, description="Patient age")
    gender: Literal["Male", "Female"] = Field(..., description="Patient gender")
    polyuria: Literal["Yes", "No"] = Field(..., description="Excessive urination")
    polydipsia: Literal["Yes", "No"] = Field(..., description="Excessive thirst")
    sudden_weight_loss: Literal["Yes", "No"] = Field(..., description="Sudden weight loss")
    weakness: Literal["Yes", "No"] = Field(..., description="Weakness")
    polyphagia: Literal["Yes", "No"] = Field(..., description="Excessive hunger")
    genital_thrush: Literal["Yes", "No"] = Field(..., description="Genital thrush")
    visual_blurring: Literal["Yes", "No"] = Field(..., description="Visual blurring")
    itching: Literal["Yes", "No"] = Field(..., description="Itching")
    irritability: Literal["Yes", "No"] = Field(..., description="Irritability")
    delayed_healing: Literal["Yes", "No"] = Field(..., description="Delayed healing")
    partial_paresis: Literal["Yes", "No"] = Field(..., description="Partial paresis")
    muscle_stiffness: Literal["Yes", "No"] = Field(..., description="Muscle stiffness")
    alopecia: Literal["Yes", "No"] = Field(..., description="Alopecia")
    obesity: Literal["Yes", "No"] = Field(..., description="Obesity")



class PredictionRequest(BaseModel):
    """Request schema for prediction endpoint"""
    patient_id: str = Field(..., description="Patient UUID")
    symptoms: Symptoms
    symptoms_text: Literal[None] | str = Field(None, description="Unstructured symptom description for LLM")
    
    @validator('patient_id')
    def validate_uuid(cls, v):
        try:
            uuid.UUID(v)
        except ValueError:
            raise ValueError('patient_id must be a valid UUID')
        return v


class TopFactor(BaseModel):
    """Top contributing factor from SHAP"""
    feature: str = Field(..., description="Feature name")
    value: str = Field(..., description="Feature value")
    impact: float = Field(..., description="SHAP impact value")


class RoutingInfo(BaseModel):
    """Routing decision and LLM analysis"""
    action: Literal["dashboard", "analyzer"]
    analysis: Literal[None] | str = None
    is_safe: bool = True
    message: Literal[None] | str = None


class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint"""
    patient_id: str
    risk_score: float = Field(..., ge=0, le=100, description="Risk probability percentage")
    risk_level: Literal["Low", "Medium", "High"] = Field(..., description="Risk category")
    confidence_score: Literal["High", "Medium", "Low"] = Field(..., description="Prediction confidence")
    top_factors: List[TopFactor] = Field(..., max_items=3, description="Top 3 contributing factors")
    routing: Literal[None] | RoutingInfo = Field(None, description="Intelligent routing decision")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    
    
class ChatRequest(BaseModel):
    """Request for context-aware chat"""
    patient_id: str = Field(..., description="Patient UUID")
    message: str = Field(..., description="User question or query")
    
    @validator('patient_id')
    def validate_uuid(cls, v):
        try:
            uuid.UUID(v)
        except ValueError:
            raise ValueError('patient_id must be a valid UUID')
        return v


class ChatResponse(BaseModel):
    """Response from chat endpoint"""
    response: str
    context_used: bool = True


class HealthStatus(BaseModel):
    """Health check response"""
    status: str = "healthy"
    model_loaded: bool
    version: str = "1.0.0"

