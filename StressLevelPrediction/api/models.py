"""
Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Dict, List

class StressLevelInput(BaseModel):
    """Input model for stress level prediction."""
    Avg_Working_Hours_Per_Day: float = Field(..., ge=0, le=24, description="Average working hours per day")
    Work_From: str = Field(..., description="Work location (Home, Office, Hybrid)")
    Work_Pressure: int = Field(..., ge=1, le=5, description="Work pressure level (1-5)")
    Manager_Support: int = Field(..., ge=1, le=5, description="Manager support level (1-5)")
    Sleeping_Habit: int = Field(..., ge=1, le=5, description="Sleeping habit quality (1-5)")
    Exercise_Habit: int = Field(..., ge=1, le=5, description="Exercise habit frequency (1-5)")
    Job_Satisfaction: int = Field(..., ge=1, le=5, description="Job satisfaction level (1-5)")
    Work_Life_Balance: str = Field(..., description="Work-life balance status (Yes, No)")
    Social_Person: int = Field(..., ge=1, le=5, description="Social person rating (1-5)")
    Lives_With_Family: str = Field(..., description="Lives with family status (Yes, No)")
    Working_State: str = Field(..., description="Working state/location")
    
    class Config:
        json_schema_extra = {
            "example": {
                "Avg_Working_Hours_Per_Day": 8.5,
                "Work_From": "Home",
                "Work_Pressure": 3,
                "Manager_Support": 4,
                "Sleeping_Habit": 4,
                "Exercise_Habit": 3,
                "Job_Satisfaction": 4,
                "Work_Life_Balance": "Yes",
                "Social_Person": 3,
                "Lives_With_Family": "Yes",
                "Working_State": "Karnataka"
            }
        }

class StressLevelPrediction(BaseModel):
    """Output model for stress level prediction."""
    stress_level: int = Field(..., description="Predicted stress level (1-5)")
    stress_label: str = Field(..., description="Human-readable stress level label")
    confidence: float = Field(..., description="Confidence score for prediction (0-1)")
    probabilities: Dict[str, float] = Field(..., description="Probabilities for each class")
    all_probabilities: List[Dict] = Field(..., description="Detailed probabilities with labels")

