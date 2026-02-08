# VitalSight ML Service

Python-based ML microservice for diabetes risk prediction with SHAP explainability.

## Setup

1. **Create virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment:**
   - Copy `.env.example` to `.env`
   - Add your Supabase Service Role Key

4. **Train the model:**
   ```bash
   python scripts/train.py
   ```

5. **Run the API:**
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

## API Endpoints

### POST /predict
Predict diabetes risk for a patient based on symptoms.

**Request:**
```json
{
  "patient_id": "uuid",
  "symptoms": {
    "age": 58,
    "gender": "Male",
    "polyuria": "Yes",
    "polydipsia": "Yes",
    ...
  }
}
```

**Response:**
```json
{
  "patient_id": "uuid",
  "risk_score": 78.5,
  "risk_level": "High",
  "confidence_score": "High",
  "top_factors": [...],
  "timestamp": "2026-02-08T10:42:21Z"
}
```

## Dataset

The model uses the diabetes risk prediction dataset with the following features:
- Age, Gender
- Symptoms: Polyuria, Polydipsia, Sudden Weight Loss, Weakness, Polyphagia
- Signs: Genital Thrush, Visual Blurring, Itching, Irritability
- Complications: Delayed Healing, Partial Paresis, Muscle Stiffness, Alopecia
- Risk Factors: Obesity
