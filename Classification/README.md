# AI Model Training Pipeline - Classification Module

A generic, modular, and reusable PyTorch Lightning pipeline for training classification models. This pipeline is fully config-driven, allowing you to train models on any tabular classification dataset by simply modifying a YAML configuration file.

## Features

- **Fully Config-Driven**: All settings (features, hyperparameters, paths) controlled via YAML files
- **Generic & Reusable**: Use the same codebase for any classification task (stress levels, sentiment, quality ratings, etc.)
- **Auto-Dimension Detection**: Automatically calculates input dimensions and number of classes from feature lists and target column
- **Categorical Target Support**: Automatically handles both integer and categorical string targets (e.g., "good", "better", "best" or "yes", "no")
- **Production-Ready**: Exports models to ONNX format with preprocessors and label encoders for easy deployment
- **PyTorch Lightning**: Built on PyTorch Lightning for scalable, professional ML training
- **Comprehensive Metrics**: Tracks Accuracy, F1-Score, Precision, and Recall (macro-averaged)

## Project Structure

```
AI-pipeline/
├── Classification/                    # Generic classification module (reusable)
│   ├── __init__.py
│   ├── cli.py                        # Custom Lightning CLI
│   ├── main.py                       # Standard CLI entry point
│   ├── mainfittest.py                # Fit+test workflow entry point
│   ├── dataset.py                    # PyTorch Dataset for tabular data
│   ├── datamodule.py                 # Lightning DataModule
│   ├── modelfactory.py               # Neural network model factory
│   ├── modelmodule.py                # Lightning Module for training
│   └── callbacks.py                  # ONNX export callback
├── StressLevelPrediction/             # Example project (config-only)
│   ├── configs/
│   │   ├── stress.yaml               # Main configuration
│   │   └── stress.local.yaml         # Local overrides
│   ├── data/
│   │   └── stress_level.csv          # Dataset
│   ├── models/                       # Output: trained models, preprocessors, label encoders
│   └── lightning_logs/               # Output: training logs
├── Regression/                        # Regression module (separate)
├── requirements.txt                  # Python dependencies
├── venv/                              # Virtual environment (create this)
├── README.md                          # Main README (regression focus)
└── README_CLASSIFICATION.md          # This file
```

## Setup Instructions

### 1. Create Virtual Environment

```bash
# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python -c "import lightning as L; import torch; print('Installation successful!')"
```

## Creating a New Classification Project

To train a model on a new dataset, follow these steps:

### Step 1: Create Project Directory Structure

```bash
mkdir YourProjectName
mkdir YourProjectName/configs
mkdir YourProjectName/data
mkdir YourProjectName/models
mkdir YourProjectName/lightning_logs
```

### Step 2: Place Your Dataset

Place your CSV file in `YourProjectName/data/your_data.csv`

**CSV Requirements:**
- Must contain feature columns (categorical and/or numeric)
- Must contain a target column (can be integers or categorical strings)
- Target column examples:
  - Integer labels: `1, 2, 3, 4, 5`
  - Categorical strings: `"good", "better", "best"` or `"yes", "no"` or `"low", "medium", "high"`
- No missing values in target column

### Step 3: Create Configuration File

Create `YourProjectName/configs/your_project.yaml`:

```yaml
# Your Classification Project Configuration
seed_everything: true

trainer:
  callbacks:
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args:
        filename: "{epoch}-{val_loss:.2f}.best"
        monitor: "val_loss"
        mode: "min"
        save_top_k: 1
        verbose: true
        save_on_train_epoch_end: false
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args:
        filename: "{epoch}.last"
        monitor: "step"
        mode: "max"
        save_top_k: 1
        verbose: true
        save_on_train_epoch_end: false
    - class_path: Classification.callbacks.ONNXExportCallback
      init_args:
        output_dir: "models"
        model_name: "your_model_name"
        input_dim: null  # Auto-detected

  logger:
    class_path: lightning.pytorch.loggers.TensorBoardLogger
    init_args:
      save_dir: "lightning_logs"
      name: "YourProjectTraining"
      default_hp_metric: false

  max_epochs: 50
  num_sanity_val_steps: 1
  check_val_every_n_epoch: 1
  log_every_n_steps: 10
  accelerator: auto
  devices: auto
  precision: 16-mixed
  default_root_dir: "lightning_logs/YourProjectTraining"

model:
  class_path: Classification.modelmodule.ModelModuleCLS
  init_args:
    lr: 0.0001
    weight_decay: 0.0
    lr_scheduler_factor: 0.5
    lr_scheduler_patience: 5
    save_dir: "models"
    name: "your_model_name"
    model:
      class_path: Classification.modelfactory.ClassificationModel
      init_args:
        input_dim: 0  # Auto-set from datamodule
        num_classes: 0  # Auto-set from datamodule
        hidden_layers: [128, 64, 32]
        dropout_rates: [0.15, 0.1, 0.05]
        activation: "relu"

optimizer: 
  class_path: torch.optim.Adam
  init_args:
    lr: 0.001
    weight_decay: 0.00001

lr_scheduler:
  class_path: torch.optim.lr_scheduler.OneCycleLR
  init_args:
    max_lr: 0.001
    pct_start: 0.1
    total_steps: 1000

data:
  class_path: Classification.datamodule.DataModuleCLS
  init_args:
    csv_path: "YourProjectName/data/your_data.csv"
    batch_size: 256
    num_workers: 0
    val_split: 0.2
    random_seed: 42
    categorical_cols:
      - column1
      - column2
      # Add your categorical column names here
    numeric_cols:
      - column3
      - column4
      # Add your numeric column names here
    target_col: "target"  # Your target column name (can be integers or strings)
    save_preprocessor: true
    preprocessor_path: "YourProjectName/models/preprocessor.joblib"

fit:
  ckpt_path: null   # Set to checkpoint path for resume training

test:
  ckpt_path: best   # Use "best" or "last" checkpoint
```

### Step 4: Create Local Override File (Optional)

Create `YourProjectName/configs/your_project.local.yaml` for local-specific settings:

```yaml
# Local configuration overrides
# Example overrides:
# trainer:
#   max_epochs: 10
#   precision: 32
# data:
#   init_args:
#     batch_size: 128
#     num_workers: 4
```

## Running the Project

**Important:** Always run commands from the project root directory (where `Classification/` folder is located).

### Option 1: Fit + Test Workflow (Recommended for Quick Testing)

**Windows (PowerShell):**
```powershell
$env:PYTHONPATH = "."; python Classification/mainfittest.py --config YourProjectName/configs/your_project.yaml --config YourProjectName/configs/your_project.local.yaml
```

**Linux/Mac:**
```bash
PYTHONPATH=. python Classification/mainfittest.py \
  --config YourProjectName/configs/your_project.yaml \
  --config YourProjectName/configs/your_project.local.yaml
```

### Option 2: Standard Lightning CLI Commands

**Training only (Windows PowerShell):**
```powershell
$env:PYTHONPATH = "."; python Classification/main.py fit --config YourProjectName/configs/your_project.yaml --config YourProjectName/configs/your_project.local.yaml
```

**Testing only (Windows PowerShell):**
```powershell
$env:PYTHONPATH = "."; python Classification/main.py test --config YourProjectName/configs/your_project.yaml --config YourProjectName/configs/your_project.local.yaml
```

### Option 3: Using Helper Scripts

**Windows (PowerShell):**
```powershell
.\run_training.ps1 --config YourProjectName/configs/your_project.yaml --config YourProjectName/configs/your_project.local.yaml
```

**Linux/Mac:**
```bash
chmod +x run_training.sh
./run_training.sh --config YourProjectName/configs/your_project.yaml --config YourProjectName/configs/your_project.local.yaml
```

### Option 4: Using Jupyter Notebook

Create a notebook (e.g., `YourProjectTrainer.ipynb`):

```python
import sys
import os
# Add project root to path
sys.path.insert(0, os.getcwd())

# Import classification modules
from Classification.modelmodule import ModelModuleCLS
from Classification.datamodule import DataModuleCLS

# Run fit and test workflow
%run Classification/mainfittest.py --config YourProjectName/configs/your_project.yaml --config YourProjectName/configs/your_project.local.yaml
```

## Target Column Types

The classification pipeline supports **two types of target columns**:

### 1. Integer Labels (e.g., 1, 2, 3, 4, 5)
- Already in numeric format
- No encoding needed
- Example: Stress levels `1, 2, 3, 4, 5`

### 2. Categorical Strings (e.g., "good", "better", "best")
- Automatically encoded to integers using `LabelEncoder`
- Original labels preserved for inference
- Example: Quality ratings `"poor", "fair", "good", "excellent"`
- Example: Binary classification `"yes", "no"` or `"positive", "negative"`

**The pipeline automatically detects and handles both types!**

## Configuration Guide

### Data Configuration

**Key Parameters:**
- `csv_path`: Path to your CSV file (relative to project root)
- `batch_size`: Batch size for training (default: 256)
- `val_split`: Validation split ratio (0.0 to 1.0, default: 0.2)
- `categorical_cols`: List of categorical feature column names
- `numeric_cols`: List of numeric feature column names
- `target_col`: Name of the target column to predict (can be integers or strings)
- `preprocessor_path`: Where to save/load the preprocessor

**Preprocessing:**
- Categorical columns: Automatically one-hot encoded (with `drop='first'`)
- Numeric columns: Automatically standardized using StandardScaler
- Target column: 
  - If integers: Used as-is
  - If strings: Automatically encoded to 0-indexed integers using LabelEncoder
- Input dimension: Automatically calculated from feature lists
- Number of classes: Automatically detected from target column

**Stratified Splitting:**
- Train/val/test splits are automatically stratified to maintain class distribution
- Works for both integer and categorical string targets

### Model Configuration

**Key Parameters:**
- `hidden_layers`: List of hidden layer sizes, e.g., `[128, 64, 32]`
- `dropout_rates`: List of dropout rates matching hidden layers, e.g., `[0.15, 0.1, 0.05]`
- `activation`: Activation function (`"relu"`, `"tanh"`, `"gelu"`, `"sigmoid"`, `"leaky_relu"`, `"elu"`)
- `input_dim`: Automatically set from datamodule (set to `0` in config)
- `num_classes`: Automatically set from datamodule (set to `0` in config)

### Trainer Configuration

**Key Parameters:**
- `max_epochs`: Number of training epochs
- `precision`: Training precision (`"16-mixed"`, `"32"`, `"bf16-mixed"`)
- `accelerator`: Hardware accelerator (`"auto"`, `"gpu"`, `"cpu"`)
- `devices`: Number of devices (`"auto"`, `1`, `[0, 1]`)

## Output Files

After training, you'll find:

1. **Models Directory** (`YourProjectName/models/`):
   - `your_model_name.onnx`: ONNX model for inference
   - `preprocessor.joblib`: Fitted preprocessor for data transformation
   - `label_encoder.joblib`: Label encoder (only if target was categorical strings)

2. **Checkpoints** (`YourProjectName/lightning_logs/YourProjectTraining/version_X/checkpoints/`):
   - `epoch-X-val_loss=Y.best.ckpt`: Best model checkpoint (based on validation loss)
   - `epoch-X.last.ckpt`: Last epoch checkpoint

3. **Training Logs** (`YourProjectName/lightning_logs/`):
   - TensorBoard logs for visualization

## Viewing Training Progress

### TensorBoard

```bash
tensorboard --logdir YourProjectName/lightning_logs
```

Then open `http://localhost:6006` in your browser.

**Metrics Tracked:**
- `train_loss`, `val_loss`, `test_loss`: CrossEntropyLoss
- `train_acc`, `val_acc`, `test_acc`: Accuracy (macro-averaged)
- `train_f1`, `val_f1`, `test_f1`: F1-Score (macro-averaged)
- `train_precision`, `val_precision`, `test_precision`: Precision (macro-averaged)
- `train_recall`, `val_recall`, `test_recall`: Recall (macro-averaged)

## Example Projects

### Stress Level Prediction

```bash
# Windows (PowerShell)
$env:PYTHONPATH = "."; python Classification/mainfittest.py --config StressLevelPrediction/configs/stress.yaml --config StressLevelPrediction/configs/stress.local.yaml
```

**Dataset:** `StressLevelPrediction/data/stress_level.csv`
- **Categorical features:** Work_From, Work_Life_Balance, Lives_With_Family, Working_State
- **Numeric features:** Avg_Working_Hours_Per_Day, Work_Pressure, Manager_Support, Sleeping_Habit, Exercise_Habit, Job_Satisfaction, Social_Person
- **Target:** Stress_Level (5 classes: 1, 2, 3, 4, 5)

### Quality Rating Prediction (Example with Categorical Target)

To create a quality rating project with categorical targets:

1. Create `QualityRating/` directory structure
2. Place your data CSV in `QualityRating/data/ratings.csv` with target column like:
   - `quality`: `["poor", "fair", "good", "excellent"]`
3. Create config file with:
   - Categorical columns: `[brand, category]`
   - Numeric columns: `[price, weight, rating_score]`
   - Target: `"quality"` (will be auto-encoded)
4. Run training as shown above

The pipeline will automatically:
- Detect that `quality` contains strings
- Encode to integers: `["poor", "fair", "good", "excellent"]` → `[0, 1, 2, 3]`
- Save the label encoder for inference
- Train the model with 4 classes

## Model Inference

After training, use the exported ONNX model, preprocessor, and label encoder:

### For Integer Targets

```python
import joblib
import onnxruntime as ort
import numpy as np
import pandas as pd

# Load preprocessor
preprocessor = joblib.load("YourProjectName/models/preprocessor.joblib")

# Load ONNX model
session = ort.InferenceSession("YourProjectName/models/your_model_name.onnx")

# Prepare input data
input_data = pd.DataFrame({
    'categorical_col': ['value1'],
    'numeric_col': [123.45],
    # ... other features
})

# Transform data
feature_cols = ['categorical_col', 'numeric_col']  # Your feature columns
transformed = preprocessor.transform(input_data[feature_cols])

# Predict
input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: transformed.astype(np.float32)})
predicted_class_idx = np.argmax(output[0][0])

print(f"Predicted class index: {predicted_class_idx}")
```

### For Categorical String Targets

```python
import joblib
import onnxruntime as ort
import numpy as np
import pandas as pd

# Load preprocessor and label encoder
preprocessor = joblib.load("YourProjectName/models/preprocessor.joblib")
label_encoder = joblib.load("YourProjectName/models/label_encoder.joblib")

# Load ONNX model
session = ort.InferenceSession("YourProjectName/models/your_model_name.onnx")

# Prepare input data
input_data = pd.DataFrame({
    'categorical_col': ['value1'],
    'numeric_col': [123.45],
    # ... other features
})

# Transform data
feature_cols = ['categorical_col', 'numeric_col']  # Your feature columns
transformed = preprocessor.transform(input_data[feature_cols])

# Predict
input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: transformed.astype(np.float32)})
predicted_class_idx = np.argmax(output[0][0])

# Decode back to original label
predicted_class = label_encoder.inverse_transform([predicted_class_idx])[0]

print(f"Predicted class: {predicted_class}")
print(f"Class index: {predicted_class_idx}")
print(f"All class names: {label_encoder.classes_.tolist()}")
```

## Troubleshooting

### Common Issues

**1. FileNotFoundError: CSV file not found**
- Check that `csv_path` in config is correct relative to project root
- Ensure CSV file exists at the specified path

**2. ValueError: Missing columns in CSV**
- Verify all column names in `categorical_cols` and `numeric_cols` exist in CSV
- Check for typos in column names

**3. Could not auto-detect input_dim or num_classes**
- Ensure datamodule can be instantiated and setup successfully
- Check that CSV file is readable and has valid data
- Verify target column contains valid class labels (integers or strings)

**4. CUDA out of memory**
- Reduce `batch_size` in data configuration
- Reduce model size (smaller `hidden_layers`)
- Use `precision: "32"` instead of `"16-mixed"`

**5. Import errors**
- Ensure virtual environment is activated
- Verify all dependencies are installed: `pip install -r requirements.txt`
- Set `PYTHONPATH=.` before running (or use helper scripts)

**6. Target column encoding issues**
- If target has mixed types, ensure all values are either all integers or all strings
- For categorical strings, ensure consistent spelling/casing
- Check that target column has no missing values

**7. Class imbalance warnings**
- If classes are highly imbalanced, consider using class weights in model config
- The pipeline uses stratified splitting to maintain class distribution

## Advanced Usage

### Custom Model Architecture

Modify `hidden_layers` and `dropout_rates` in config:

```yaml
model:
  init_args:
    model:
      init_args:
        hidden_layers: [256, 128, 64, 32]  # Deeper network
        dropout_rates: [0.2, 0.15, 0.1, 0.05]
        activation: "gelu"
```

### Class Weights for Imbalanced Datasets

If your dataset has imbalanced classes, you can use class weights:

```python
# Calculate class weights (example)
from sklearn.utils.class_weight import compute_class_weight
import torch

# In your config or code
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.FloatTensor(class_weights)

# Then in model config:
model:
  init_args:
    class_weights: [0.5, 1.0, 1.5, 2.0, 1.0]  # Example weights for 5 classes
```

### Hyperparameter Tuning

Override hyperparameters in local config:

```yaml
# your_project.local.yaml
model:
  init_args:
    lr: 0.0005
optimizer:
  init_args:
    lr: 0.002
    weight_decay: 0.0001
data:
  init_args:
    batch_size: 512
```

### Resume Training

```bash
# Windows (PowerShell)
$env:PYTHONPATH = "."; python Classification/main.py fit \
  --config YourProjectName/configs/your_project.yaml \
  --fit.ckpt_path "YourProjectName/lightning_logs/YourProjectTraining/version_0/checkpoints/epoch-10.last.ckpt"
```

## Key Differences from Regression

| Feature | Regression | Classification |
|---------|-----------|---------------|
| **Output** | Continuous value (float) | Class label (integer) |
| **Loss Function** | MSE (Mean Squared Error) | CrossEntropyLoss |
| **Metrics** | MAE (Mean Absolute Error) | Accuracy, F1, Precision, Recall |
| **Model Output** | Single value | Logits (num_classes values) |
| **Target Encoding** | None (already numeric) | Auto-encodes categorical strings |
| **Output Files** | Model + Preprocessor | Model + Preprocessor + Label Encoder |

## Best Practices

1. **Stratified Splitting**: The pipeline automatically uses stratified splitting to maintain class distribution
2. **Class Balance**: Check class distribution before training. Use class weights if highly imbalanced
3. **Target Encoding**: Let the pipeline handle encoding automatically - it works for both integers and strings
4. **Metrics**: Monitor F1-Score in addition to Accuracy, especially for imbalanced datasets
5. **Validation**: Use validation set to tune hyperparameters and prevent overfitting
6. **Inference**: Always use the saved label encoder when target was categorical strings

## Contributing

This is a generic pipeline designed to be extended. To add new features:

1. Modify files in `Classification/` for generic improvements
2. Keep project-specific code in project directories (configs only)
3. Follow the config-driven approach
4. Test with both integer and categorical string targets

## License

[Add your license here]

## Support

For issues or questions, please check:
1. Configuration file syntax
2. CSV file format and column names
3. Target column type (integers or categorical strings)
4. Dependencies installation
5. TensorBoard logs for training insights

```
# Install dependencies
pip install fastapi uvicorn jinja2 python-multipart onnxruntime

# Run the server
python -m StressLevelPrediction.api.main

# Or using uvicorn directly
uvicorn StressLevelPrediction.api.main:app --reload --host 0.0.0.0 --port 8000
uvicorn StressLevelPrediction.api.main:app --reload  --port 8000
```