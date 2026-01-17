"""
ONNX model predictor for stress level prediction.
"""

import onnxruntime as ort
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Any, Optional

class StressLevelPredictor:
    """Predictor class for stress level using ONNX model."""
    
    def __init__(
        self,
        model_path: Path,
        preprocessor_path: Path,
        label_encoder_path: Optional[Path] = None
    ):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.label_encoder_path = label_encoder_path
        self.session = None
        self.preprocessor = None
        self.label_encoder = None
        
    def load(self):
        """Load ONNX model, preprocessor, and label encoder."""
        # Load ONNX model
        self.session = ort.InferenceSession(str(self.model_path))
        
        # Load preprocessor
        self.preprocessor = joblib.load(self.preprocessor_path)
        
        # Load label encoder if available
        if self.label_encoder_path and self.label_encoder_path.exists():
            self.label_encoder = joblib.load(self.label_encoder_path)
    
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make prediction on input data.
        
        Args:
            input_data: Dictionary with feature values
            
        Returns:
            Dictionary with prediction results
        """
        if self.session is None or self.preprocessor is None:
            raise RuntimeError("Predictor not loaded. Call load() first.")
        
        # Convert to DataFrame
        df = pd.DataFrame([input_data])
        
        # Get feature columns (categorical + numeric) - must match training config
        categorical_cols = ['Work_From', 'Work_Life_Balance', 'Lives_With_Family', 'Working_State']
        numeric_cols = [
            'Avg_Working_Hours_Per_Day', 'Work_Pressure', 'Manager_Support',
            'Sleeping_Habit', 'Exercise_Habit', 'Job_Satisfaction', 'Social_Person'
        ]
        feature_cols = categorical_cols + numeric_cols
        
        # Preprocess
        features = df[feature_cols]
        transformed = self.preprocessor.transform(features)
        
        # ONNX inference
        input_name = self.session.get_inputs()[0].name
        output = self.session.run(None, {input_name: transformed.astype(np.float32)})
        logits = output[0][0]  # Get first (and only) sample
        
        # Get predicted class (0-indexed from model)
        predicted_class_idx = int(np.argmax(logits))
        
        # Apply softmax to get probabilities
        exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
        probabilities = exp_logits / np.sum(exp_logits)
        confidence = float(probabilities[predicted_class_idx])
        
        # Map back to original label (1-5 scale)
        if self.label_encoder:
            # Reverse transform: 0-indexed -> original label
            predicted_label = int(self.label_encoder.inverse_transform([predicted_class_idx])[0])
        else:
            # If no label encoder, assume model outputs 0-4, map to 1-5
            predicted_label = predicted_class_idx + 1
        
        # Get class probabilities for all classes
        prob_dict = {
            f"class_{i+1}": float(prob) 
            for i, prob in enumerate(probabilities)
        }
        
        # Stress level labels
        stress_labels = {
            1: "Low",
            2: "Low-Medium", 
            3: "Medium",
            4: "Medium-High",
            5: "High"
        }
        
        return {
            "stress_level": predicted_label,
            "stress_label": stress_labels.get(predicted_label, "Unknown"),
            "confidence": confidence,
            "probabilities": prob_dict,
            "all_probabilities": [
                {
                    "level": i+1,
                    "label": stress_labels.get(i+1, "Unknown"),
                    "probability": float(prob)
                }
                for i, prob in enumerate(probabilities)
            ]
        }

