"""
FastAPI application for Stress Level Prediction.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import pandas as pd
from typing import Optional

from .predictor import StressLevelPredictor
from .models import StressLevelInput, StressLevelPrediction

# Global predictor instance
predictor: Optional[StressLevelPredictor] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    # Startup
    global predictor
    try:
        # Setup paths - models are in StressLevelPrediction/models/
        STRESS_LEVEL_DIR = Path(__file__).resolve().parent.parent
        MODELS_DIR = STRESS_LEVEL_DIR / "models"
        
        predictor = StressLevelPredictor(
            model_path=MODELS_DIR / "stress_level_model.onnx",
            preprocessor_path=MODELS_DIR / "preprocessor.joblib",
            label_encoder_path=MODELS_DIR / "label_encoder.joblib"
        )
        predictor.load()
        print("✓ Models loaded successfully")
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        raise
    
    yield  # Application runs here
    
    # Shutdown (cleanup if needed)
    pass

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Stress Level Prediction API",
    description="API for predicting employee stress levels using machine learning",
    version="1.0.0",
    lifespan=lifespan
)

# Setup paths
STRESS_LEVEL_DIR = Path(__file__).resolve().parent.parent
API_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = API_DIR / "templates"
STATIC_DIR = API_DIR / "static"
DATA_DIR = STRESS_LEVEL_DIR / "data"

# Mount static files and setup templates
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

def get_dropdown_options():
    """Get unique values for dropdown options from dataset."""
    dataset_path = DATA_DIR / "stress_level.csv"
    if not dataset_path.exists():
        # Return default values if dataset not found
        return {
            "work_from_options": ["Home", "Office", "Hybrid"],
            "work_life_balance_options": ["Yes", "No"],
            "lives_with_family_options": ["Yes", "No"],
            "working_state_options": ["Karnataka", "Pune", "Delhi", "Hyderabad"]
        }
    
    df = pd.read_csv(dataset_path)
    return {
        "work_from_options": sorted(df["Work_From"].unique().tolist()),
        "work_life_balance_options": sorted(df["Work_Life_Balance"].unique().tolist()),
        "lives_with_family_options": sorted(df["Lives_With_Family"].unique().tolist()),
        "working_state_options": sorted(df["Working_State"].unique().tolist()),
    }

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the main prediction form."""
    options = get_dropdown_options()
    context = {
        "request": request,
        **options
    }
    return templates.TemplateResponse("index.html", context)

@app.post("/predict", response_class=HTMLResponse)
async def predict_web(
    request: Request,
    avg_working_hours: float = Form(...),
    work_from: str = Form(...),
    work_pressure: int = Form(...),
    manager_support: int = Form(...),
    sleeping_habit: int = Form(...),
    exercise_habit: int = Form(...),
    job_satisfaction: int = Form(...),
    work_life_balance: str = Form(...),
    social_person: int = Form(...),
    lives_with_family: str = Form(...),
    working_state: str = Form(...),
):
    """Handle form submission and return prediction."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please try again later.")
    
    try:
        # Create input dict
        input_data = {
            "Avg_Working_Hours_Per_Day": avg_working_hours,
            "Work_From": work_from,
            "Work_Pressure": work_pressure,
            "Manager_Support": manager_support,
            "Sleeping_Habit": sleeping_habit,
            "Exercise_Habit": exercise_habit,
            "Job_Satisfaction": job_satisfaction,
            "Work_Life_Balance": work_life_balance,
            "Social_Person": social_person,
            "Lives_With_Family": lives_with_family,
            "Working_State": working_state,
        }
        
        # Get prediction
        prediction = predictor.predict(input_data)
        
        # Get dropdown options
        options = get_dropdown_options()
        
        context = {
            "request": request,
            "prediction": prediction,
            "input_data": input_data,
            **options
        }
        return templates.TemplateResponse("index.html", context)
    
    except Exception as e:
        options = get_dropdown_options()
        context = {
            "request": request,
            "error": str(e),
            **options
        }
        return templates.TemplateResponse("index.html", context)

@app.post("/api/predict", response_model=StressLevelPrediction)
async def predict_api(input_data: StressLevelInput):
    """REST API endpoint for JSON predictions."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please try again later.")
    
    try:
        prediction = predictor.predict(input_data.dict())
        return StressLevelPrediction(**prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

