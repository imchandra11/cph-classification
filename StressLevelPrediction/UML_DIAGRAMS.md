# Stress Level Prediction API - UML Diagrams & Architecture Documentation

This document provides comprehensive UML diagrams and architectural documentation for the Stress Level Prediction API, showing how the ONNX model is integrated with FastAPI and how the web interface connects to the inference pipeline.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Class Diagrams](#class-diagrams)
3. [Sequence Diagrams](#sequence-diagrams)
4. [Component Diagrams](#component-diagrams)
5. [Activity Diagrams](#activity-diagrams)
6. [Class-Level Details](#class-level-details)

---

## Architecture Overview

The Stress Level Prediction API is a production-ready inference service that serves a trained classification model via FastAPI. It predicts employee stress levels (1-5 scale) based on tabular employee data.

**Key Components**:
- **FastAPI Application**: HTTP server with web UI and REST API
- **ONNX Runtime**: Efficient model inference engine
- **Predictor Service**: Handles data preprocessing and ONNX inference
- **Preprocessor**: sklearn ColumnTransformer (one-hot encoding, scaling)
- **Label Encoder**: Maps model output (0-4) back to stress levels (1-5)
- **Web Interface**: Jinja2-templated HTML with form inputs

**Architecture Layers**:
1. **Presentation Layer**: Web UI (HTML/CSS/JS) and REST API
2. **Application Layer**: FastAPI routes and request handling
3. **Service Layer**: Predictor class (preprocessing + inference)
4. **Model Layer**: ONNX Runtime with loaded model
5. **Data Layer**: Model files (ONNX, preprocessor, label encoder)

**Key Differences from Image Classification**:
- **Input Type**: Tabular form data (not images)
- **Preprocessing**: sklearn ColumnTransformer (not image transforms)
- **Data Source**: Form fields (not file upload)
- **Output Mapping**: Label encoder maps 0-indexed to 1-5 stress levels

---

## Class Diagrams

### Complete Class Diagram

```mermaid
classDiagram
    class FastAPI {
        +app: FastAPI
        +lifespan: asynccontextmanager
        +mount(static_files)
        +get(path)
        +post(path)
    }
    
    class StressLevelApp {
        -app: FastAPI
        -predictor: Optional[StressLevelPredictor]
        -templates: Jinja2Templates
        -data_dir: Path
        +lifespan(app: FastAPI)
        +get_dropdown_options() dict
        +home(request: Request) HTMLResponse
        +predict_web(request, form_fields) HTMLResponse
        +predict_api(input_data: StressLevelInput) StressLevelPrediction
        +health_check() dict
    }
    
    class StressLevelPredictor {
        -model_path: Path
        -preprocessor_path: Path
        -label_encoder_path: Optional[Path]
        -session: Optional[InferenceSession]
        -preprocessor: Optional[ColumnTransformer]
        -label_encoder: Optional[LabelEncoder]
        +__init__(model_path, preprocessor_path, label_encoder_path)
        +load()
        +predict(input_data: Dict) dict
        -_map_to_stress_label(level: int) str
    }
    
    class StressLevelInput {
        +Avg_Working_Hours_Per_Day: float
        +Work_From: str
        +Work_Pressure: int
        +Manager_Support: int
        +Sleeping_Habit: int
        +Exercise_Habit: int
        +Job_Satisfaction: int
        +Work_Life_Balance: str
        +Social_Person: int
        +Lives_With_Family: str
        +Working_State: str
    }
    
    class StressLevelPrediction {
        +stress_level: int
        +stress_label: str
        +confidence: float
        +probabilities: Dict[str, float]
        +all_probabilities: List[Dict]
    }
    
    class ColumnTransformer {
        +transform(X) ndarray
        -categorical_transformer: OneHotEncoder
        -numeric_transformer: StandardScaler
    }
    
    class LabelEncoder {
        +inverse_transform(indices) list
        +classes_: ndarray
    }
    
    class InferenceSession {
        +get_inputs() list
        +run(output_names, input_dict) list
    }
    
    class DataFrame {
        +pd.DataFrame()
    }
    
    %% Relationships
    StressLevelApp --> FastAPI : uses
    StressLevelApp --> StressLevelPredictor : contains
    StressLevelApp --> Jinja2Templates : uses
    StressLevelApp --> StressLevelInput : accepts
    StressLevelApp --> StressLevelPrediction : returns
    StressLevelApp --> DataFrame : reads CSV for dropdowns
    
    StressLevelPredictor --> InferenceSession : uses
    StressLevelPredictor --> ColumnTransformer : uses
    StressLevelPredictor --> LabelEncoder : uses
    StressLevelPredictor --> DataFrame : creates for preprocessing
    
    StressLevelInput --> StressLevelPredictor : converted to dict
    StressLevelPrediction --> StressLevelInput : validates response
    
    InferenceSession ..> ONNXModel : loads
    ColumnTransformer ..> joblib : loads from
    LabelEncoder ..> joblib : loads from
    
    note for StressLevelApp "Endpoints:\n- GET / (web form)\n- POST /predict (form submission)\n- POST /api/predict (REST API)\n- GET /health"
    
    note for StressLevelPredictor "Features:\n- 4 Categorical (one-hot encoded)\n- 7 Numeric (standardized)\n- Output: 5 stress levels (1-5)"
```

### Data Preprocessing Flow

```mermaid
classDiagram
    class InputData {
        <<Dictionary>>
        +Avg_Working_Hours_Per_Day: float
        +Work_From: str
        +Work_Pressure: int
        ...
    }
    
    class DataFrame {
        +pd.DataFrame([input_data])
        +columns: list
    }
    
    class ColumnTransformer {
        +transform(features) ndarray
    }
    
    class CategoricalTransformer {
        +OneHotEncoder
        +fit_transform()
    }
    
    class NumericTransformer {
        +StandardScaler
        +transform()
    }
    
    class TransformedFeatures {
        <<numpy array>>
        +shape: [1, num_features]
        +dtype: float32
    }
    
    InputData --> DataFrame : Convert to DataFrame
    DataFrame --> ColumnTransformer : Extract features
    ColumnTransformer --> CategoricalTransformer : Transform categorical
    ColumnTransformer --> NumericTransformer : Transform numeric
    CategoricalTransformer --> TransformedFeatures : One-hot encoded
    NumericTransformer --> TransformedFeatures : Standardized
    TransformedFeatures --> ONNXModel : Input array
```

---

## Sequence Diagrams

### Web Form Prediction Flow

```mermaid
sequenceDiagram
    participant User
    participant Browser
    participant FastAPI as FastAPI App
    participant Route as /predict Route
    participant Predictor as StressLevelPredictor
    participant DataFrame as pandas DataFrame
    participant Preprocessor as ColumnTransformer
    participant ONNX as ONNX Runtime
    participant Model as ONNX Model
    participant LabelEnc as Label Encoder
    participant Template as Jinja2 Template
    participant Response as HTML Response

    User->>Browser: Fill form & submit
    Browser->>FastAPI: POST /predict (form-data)
    
    FastAPI->>Route: predict_web(request, form_fields)
    
    Route->>Route: Build input_data dict from form
    Note over Route: 11 form fields:<br/>- Numeric (hours, ratings 1-5)<br/>- Categorical (dropdowns)
    
    Route->>Predictor: predict(input_data)
    
    Predictor->>DataFrame: pd.DataFrame([input_data])
    DataFrame-->>Predictor: df with 1 row
    
    Predictor->>Predictor: Extract feature columns
    Note over Predictor: Categorical: Work_From,<br/>Work_Life_Balance,<br/>Lives_With_Family,<br/>Working_State<br/>Numeric: 7 numeric columns
    
    Predictor->>Preprocessor: transform(features)
    Preprocessor->>Preprocessor: One-hot encode categorical
    Preprocessor->>Preprocessor: Standardize numeric
    Preprocessor-->>Predictor: transformed array [1, num_features]
    
    Predictor->>ONNX: session.run(None, {input_name: array})
    ONNX->>Model: Forward pass
    Model-->>ONNX: logits [5]
    ONNX-->>Predictor: logits
    
    Predictor->>Predictor: Apply softmax (numerically stable)
    Predictor->>Predictor: Get predicted class index (0-4)
    Predictor->>Predictor: Get confidence
    
    Predictor->>LabelEnc: inverse_transform([predicted_idx])
    LabelEnc-->>Predictor: stress_level (1-5)
    
    Predictor->>Predictor: Map to stress label<br/>(Low, Low-Medium, etc.)
    Predictor->>Predictor: Build all_probabilities list
    Predictor-->>Route: prediction dict
    
    Route->>Route: get_dropdown_options() [from CSV]
    Route->>Template: TemplateResponse("index.html", context)
    Template->>Template: Render HTML with prediction
    Template-->>Route: HTML content
    Route->>Response: HTMLResponse(html)
    Response-->>Browser: HTML with results
    Browser->>User: Display prediction & form
```

### REST API Prediction Flow

```mermaid
sequenceDiagram
    participant Client
    participant FastAPI as FastAPI App
    participant Route as /api/predict Route
    participant Pydantic as StressLevelInput
    participant Predictor as StressLevelPredictor
    participant Preprocessor as ColumnTransformer
    participant ONNX as ONNX Runtime
    participant Response as JSON Response

    Client->>FastAPI: POST /api/predict (JSON)
    
    FastAPI->>Route: predict_api(input_data: StressLevelInput)
    Route->>Pydantic: Validate JSON input
    Pydantic->>Pydantic: Check field types & constraints
    Pydantic-->>Route: Validated StressLevelInput
    
    Route->>Predictor: predict(input_data.dict())
    
    Predictor->>Predictor: Convert dict to DataFrame
    Predictor->>Preprocessor: transform(features)
    Preprocessor-->>Predictor: transformed array
    
    Predictor->>ONNX: session.run(...)
    ONNX-->>Predictor: logits
    
    Predictor->>Predictor: Process logits & map labels
    Predictor-->>Route: prediction dict
    
    Route->>Pydantic: StressLevelPrediction(**prediction)
    Pydantic->>Pydantic: Validate & serialize
    Pydantic-->>Route: StressLevelPrediction object
    Route->>Response: JSONResponse(prediction)
    Response-->>Client: JSON with stress level prediction
```

### Application Startup Sequence

```mermaid
sequenceDiagram
    participant Server
    participant FastAPI as FastAPI App
    participant Lifespan as Lifespan Handler
    participant Predictor as StressLevelPredictor
    participant FileSystem as File System
    participant ONNX as ONNX Runtime
    participant Joblib as joblib
    participant Preprocessor as ColumnTransformer
    participant LabelEnc as Label Encoder

    Server->>FastAPI: Start application
    FastAPI->>Lifespan: lifespan(app) [startup]
    
    Lifespan->>Lifespan: Setup paths (models directory)
    Lifespan->>Predictor: StressLevelPredictor(...)
    Predictor->>Predictor: Store paths
    
    Lifespan->>Predictor: load()
    
    Predictor->>FileSystem: Check model.onnx exists
    FileSystem-->>Predictor: model.onnx
    
    Predictor->>ONNX: InferenceSession(model_path)
    ONNX->>FileSystem: Load ONNX model
    FileSystem-->>ONNX: Model bytes
    ONNX->>ONNX: Initialize session
    ONNX-->>Predictor: session object
    
    Predictor->>FileSystem: Check preprocessor.joblib exists
    FileSystem-->>Predictor: preprocessor.joblib
    
    Predictor->>Joblib: load(preprocessor_path)
    Joblib->>FileSystem: Read file
    FileSystem-->>Joblib: Serialized ColumnTransformer
    Joblib-->>Predictor: ColumnTransformer object
    
    Predictor->>FileSystem: Check label_encoder.joblib exists
    FileSystem-->>Predictor: label_encoder.joblib (optional)
    
    Predictor->>Joblib: load(label_encoder_path)
    Joblib->>FileSystem: Read file
    FileSystem-->>Joblib: Serialized LabelEncoder
    Joblib-->>Predictor: LabelEncoder object
    
    Predictor-->>Lifespan: Load complete
    
    Lifespan->>Lifespan: Print success message
    Lifespan-->>FastAPI: Application ready
    FastAPI-->>Server: Server running on port 8000
```

### Data Preprocessing Sequence

```mermaid
sequenceDiagram
    participant Predictor
    participant DataFrame as pandas DataFrame
    participant Preprocessor as ColumnTransformer
    participant OneHot as OneHotEncoder
    participant Scaler as StandardScaler
    participant NumPy as NumPy Array
    participant ONNX as ONNX Runtime

    Predictor->>DataFrame: pd.DataFrame([input_data])
    DataFrame-->>Predictor: df with 11 columns
    
    Predictor->>Predictor: Extract feature columns
    Note over Predictor: Categorical: 4 columns<br/>Numeric: 7 columns
    
    Predictor->>Preprocessor: transform(df[feature_cols])
    
    Preprocessor->>OneHot: Transform categorical columns
    OneHot->>OneHot: One-hot encode categories
    OneHot-->>Preprocessor: One-hot encoded arrays
    
    Preprocessor->>Scaler: Transform numeric columns
    Scaler->>Scaler: Standardize (mean=0, std=1)
    Scaler-->>Preprocessor: Standardized arrays
    
    Preprocessor->>Preprocessor: Concatenate categorical + numeric
    Preprocessor-->>Predictor: Transformed array [1, num_features]
    
    Predictor->>NumPy: array.astype(np.float32)
    NumPy-->>Predictor: float32 array
    
    Predictor->>ONNX: session.run(None, {input_name: array})
    ONNX-->>Predictor: logits [5]
```

---

## Component Diagrams

### System Architecture

```mermaid
graph TB
    subgraph "Client Layer"
        Browser[Web Browser<br/>Form Interface]
        RESTClient[REST API Client<br/>curl/Postman/etc]
    end
    
    subgraph "HTTP Layer"
        FastAPI[FastAPI Application<br/>HTTP Server]
        Uvicorn[Uvicorn ASGI Server]
    end
    
    subgraph "Route Layer"
        WebRoute[Web Routes<br/>GET /, POST /predict]
        APIRoute[API Routes<br/>POST /api/predict<br/>GET /health]
    end
    
    subgraph "Application Layer"
        DropdownLoader[Dropdown Options Loader<br/>Read from CSV]
        Predictor[StressLevelPredictor<br/>Inference Service]
        TemplateEngine[Jinja2 Templates<br/>HTML Rendering]
    end
    
    subgraph "Service Layer"
        DataFrameBuilder[DataFrame Builder<br/>Convert dict to DataFrame]
        Preprocessor[ColumnTransformer<br/>One-hot + Standardize]
        ONNXRuntime[ONNX Runtime<br/>Inference Engine]
        LabelMapper[Label Mapper<br/>0-4 to 1-5 + Labels]
    end
    
    subgraph "Model Layer"
        ONNXModel[ONNX Model<br/>stress_level_model.onnx]
        PreprocessorFile[Preprocessor<br/>preprocessor.joblib]
        LabelEncoder[Label Encoder<br/>label_encoder.joblib]
    end
    
    subgraph "Data Layer"
        ModelFiles[Model Files<br/>File System]
        DatasetCSV[Dataset CSV<br/>stress_level.csv]
        StaticFiles[Static Files<br/>CSS]
        TemplateFiles[Template Files<br/>HTML Templates]
    end
    
    %% Client to HTTP
    Browser --> FastAPI
    RESTClient --> FastAPI
    FastAPI --> Uvicorn
    
    %% HTTP to Routes
    FastAPI --> WebRoute
    FastAPI --> APIRoute
    
    %% Routes to Application
    WebRoute --> DropdownLoader
    WebRoute --> Predictor
    WebRoute --> TemplateEngine
    APIRoute --> Predictor
    
    %% Application to Service
    Predictor --> DataFrameBuilder
    Predictor --> Preprocessor
    Predictor --> ONNXRuntime
    Predictor --> LabelMapper
    
    %% Service to Model
    ONNXRuntime --> ONNXModel
    Preprocessor --> PreprocessorFile
    LabelMapper --> LabelEncoder
    
    %% Model to Data
    ONNXModel --> ModelFiles
    PreprocessorFile --> ModelFiles
    LabelEncoder --> ModelFiles
    
    %% Dropdown loader
    DropdownLoader --> DatasetCSV
    
    %% Static/Template Files
    FastAPI --> StaticFiles
    TemplateEngine --> TemplateFiles
    
    style FastAPI fill:#667eea,color:#fff
    style Predictor fill:#764ba2,color:#fff
    style ONNXRuntime fill:#f093fb,color:#fff
    style ONNXModel fill:#ff6b6b,color:#fff
    style Preprocessor fill:#4ecdc4,color:#fff
```

### Request Processing Pipeline

```mermaid
graph LR
    A[HTTP Request] --> B{Request Type?}
    
    B -->|Web Form| C[GET /]
    B -->|Form Submit| D[POST /predict]
    B -->|REST API| E[POST /api/predict]
    B -->|Health| F[GET /health]
    
    C --> G[Load Dropdown Options<br/>from CSV]
    G --> H[Render Template<br/>with form]
    H --> I[HTML Response]
    
    D --> J[Extract Form Fields]
    E --> K[Validate JSON<br/>with Pydantic]
    
    J --> L[Build Input Dict]
    K --> L
    
    L --> M[Convert to DataFrame]
    M --> N[Extract Features]
    N --> O[Preprocess<br/>One-hot + Scale]
    O --> P[ONNX Inference]
    P --> Q[Postprocess<br/>Softmax + Label Mapping]
    
    Q --> R{Response Type?}
    R -->|Web| S[Render Template<br/>with results]
    R -->|API| T[Serialize JSON]
    
    S --> U[HTML Response]
    T --> V[JSON Response]
    
    F --> W[Health Status]
    W --> V
    
    style P fill:#ff6b6b,color:#fff
    style Q fill:#764ba2,color:#fff
    style O fill:#4ecdc4,color:#fff
```

---

## Activity Diagrams

### Complete Prediction Workflow

```mermaid
flowchart TD
    Start([User Submits Form]) --> Receive[Receive HTTP Request]
    Receive --> ExtractFields[Extract Form Fields]
    
    ExtractFields --> ValidateFields{Validate Fields?}
    ValidateFields -- Invalid --> Error1[Return Error: Invalid Input]
    ValidateFields -- Valid --> BuildDict[Build Input Dictionary]
    
    BuildDict --> ConvertDF[Convert to pandas DataFrame]
    ConvertDF --> ExtractFeatures[Extract Feature Columns]
    
    ExtractFeatures --> SeparateColumns{Separate Columns?}
    SeparateColumns -- Categorical --> OneHot[One-Hot Encode<br/>4 categorical columns]
    SeparateColumns -- Numeric --> Standardize[Standardize<br/>7 numeric columns]
    
    OneHot --> Concatenate[Concatenate Features]
    Standardize --> Concatenate
    
    Concatenate --> ToFloat32[Convert to float32]
    ToFloat32 --> ONNXInference[ONNX Runtime Inference]
    
    ONNXInference --> GetLogits[Get Logits Array 5]
    GetLogits --> Softmax[Apply Softmax]
    Softmax --> GetPredictedIdx[Get Predicted Index 0-4]
    GetPredictedIdx --> MapToLevel[Map to Stress Level 1-5]
    
    MapToLevel --> CheckEncoder{Label Encoder?}
    CheckEncoder -- Yes --> InverseTransform[inverse_transform idx]
    CheckEncoder -- No --> AddOne[Add 1 to index]
    
    InverseTransform --> GetLabel
    AddOne --> GetLabel[Get Stress Label<br/>Low, Medium, High, etc.]
    
    GetLabel --> CalculateConf[Calculate Confidence]
    CalculateConf --> BuildProbs[Build Probability Lists]
    BuildProbs --> BuildResponse[Build Response Dict]
    
    BuildResponse --> CheckType{Response Type?}
    CheckType -- Web UI --> LoadDropdowns[Load Dropdown Options]
    CheckType -- REST API --> SerializeJSON[Serialize to JSON]
    
    LoadDropdowns --> RenderTemplate[Render Jinja2 Template]
    RenderTemplate --> HTMLResponse[Return HTML Response]
    
    SerializeJSON --> ValidatePydantic[Validate with Pydantic]
    ValidatePydantic --> JSONResponse[Return JSON Response]
    
    HTMLResponse --> End([User Sees Results])
    JSONResponse --> End
    Error1 --> End
    
    style Start fill:#90EE90
    style End fill:#FFB6C1
    style ONNXInference fill:#87CEEB
    style Softmax fill:#FFD700
    style OneHot fill:#4ecdc4
    style Standardize fill:#4ecdc4
```

### Application Startup Workflow

```mermaid
flowchart TD
    Start([Server Starts]) --> InitFastAPI[Initialize FastAPI App]
    InitFastAPI --> RegisterLifespan[Register Lifespan Handler]
    RegisterLifespan --> Startup[Lifespan Startup]
    
    Startup --> SetupPaths[Setup Model Paths]
    SetupPaths --> CheckModel{Model File Exists?}
    
    CheckModel -->|No| Error1[Raise FileNotFoundError]
    CheckModel -->|Yes| CheckPreprocessor{Preprocessor Exists?}
    
    CheckPreprocessor -->|No| Error2[Raise FileNotFoundError]
    CheckPreprocessor -->|Yes| CreatePredictor[Create Predictor Instance]
    
    CreatePredictor --> LoadONNX[Load ONNX Model]
    LoadONNX --> InitSession[Initialize ONNX Session]
    InitSession --> LoadPreprocessor[Load ColumnTransformer]
    LoadPreprocessor --> CheckEncoder{Label Encoder Exists?}
    
    CheckEncoder -->|Yes| LoadEncoder[Load Label Encoder]
    CheckEncoder -->|No| SkipEncoder[Skip Label Encoder<br/>Use default mapping]
    
    LoadEncoder --> Ready
    SkipEncoder --> Ready[Predictor Ready]
    
    Ready --> RegisterRoutes[Register FastAPI Routes]
    RegisterRoutes --> MountStatic[Mount Static Files]
    MountStatic --> MountTemplates[Setup Jinja2 Templates]
    MountTemplates --> StartServer[Start Uvicorn Server]
    StartServer --> Running([Server Running on Port 8000])
    
    Error1 --> End([Startup Failed])
    Error2 --> End
    
    style Start fill:#90EE90
    style Running fill:#90EE90
    style End fill:#FFB6C1
    style LoadONNX fill:#87CEEB
    style LoadPreprocessor fill:#4ecdc4
```

### Form Data Preprocessing Workflow

```mermaid
flowchart LR
    A[Form Input Dict] --> B[pd.DataFrame]
    B --> C[Extract Feature Columns]
    
    C --> D{Categorical?}
    C --> E{Numeric?}
    
    D -->|Yes| F[OneHotEncoder]
    D -->|No| G[Skip]
    
    E -->|Yes| H[StandardScaler]
    E -->|No| G
    
    F --> I[One-Hot Encoded Array]
    H --> J[Standardized Array]
    G --> K[Concatenate]
    
    I --> K
    J --> K
    
    K --> L[Combined Feature Array<br/>1, num_features]
    L --> M[float32 Conversion]
    M --> N[ONNX Input]
    
    style F fill:#4ecdc4,color:#fff
    style H fill:#4ecdc4,color:#fff
    style N fill:#ff6b6b,color:#fff
```

---

## Class-Level Details

### StressLevelApp (main.py)

**Purpose**: FastAPI application that serves the stress level prediction web interface and REST API.

**Key Responsibilities**:
- Initialize FastAPI app with lifespan events
- Register HTTP routes (web UI and REST API)
- Handle form submissions
- Load dropdown options from dataset CSV
- Coordinate between predictor and templates
- Serve static files and templates

**Key Attributes**:
- `app`: FastAPI application instance
- `predictor`: Global predictor instance (loaded at startup)
- `templates`: Jinja2 template engine
- `data_dir`: Directory containing dataset CSV

**Key Methods**:
- `lifespan()`: Startup/shutdown handler (loads model)
- `get_dropdown_options()`: Reads CSV to get unique values for dropdowns
- `home()`: Renders main form page (GET /)
- `predict_web()`: Handles form submission (POST /predict)
- `predict_api()`: REST API endpoint (POST /api/predict)
- `health_check()`: Health check endpoint (GET /health)

**Endpoints**:
1. `GET /` - Web interface home page (form)
2. `POST /predict` - Web form submission (returns HTML)
3. `POST /api/predict` - REST API (returns JSON)
4. `GET /health` - Health check

**Form Fields** (11 inputs):
- `Avg_Working_Hours_Per_Day`: float (0-24)
- `Work_From`: str (dropdown: Home/Office/Hybrid)
- `Work_Pressure`: int (slider: 1-5)
- `Manager_Support`: int (slider: 1-5)
- `Sleeping_Habit`: int (slider: 1-5)
- `Exercise_Habit`: int (slider: 1-5)
- `Job_Satisfaction`: int (slider: 1-5)
- `Work_Life_Balance`: str (dropdown: Yes/No)
- `Social_Person`: int (slider: 1-5)
- `Lives_With_Family`: str (dropdown: Yes/No)
- `Working_State`: str (dropdown: varies)

---

### StressLevelPredictor

**Purpose**: Service class that handles data preprocessing and ONNX model inference.

**Key Responsibilities**:
- Load ONNX model, preprocessor, and label encoder at startup
- Preprocess input data (one-hot encoding, standardization)
- Run ONNX inference
- Postprocess results (softmax, label mapping, stress level labels)
- Map model output (0-4) to stress levels (1-5)

**Key Attributes**:
- `session`: ONNX Runtime InferenceSession
- `preprocessor`: sklearn ColumnTransformer (fitted)
- `label_encoder`: sklearn LabelEncoder (optional, for mapping)
- `model_path`: Path to ONNX model
- `preprocessor_path`: Path to preprocessor joblib file
- `label_encoder_path`: Path to label encoder joblib file

**Key Methods**:
- `load()`: Loads ONNX model, preprocessor, and label encoder
- `predict()`: Main prediction method (returns dict with results)
- `_map_to_stress_label()`: Maps stress level (1-5) to human-readable label

**Feature Columns**:
- **Categorical (4)**: `Work_From`, `Work_Life_Balance`, `Lives_With_Family`, `Working_State`
- **Numeric (7)**: `Avg_Working_Hours_Per_Day`, `Work_Pressure`, `Manager_Support`, `Sleeping_Habit`, `Exercise_Habit`, `Job_Satisfaction`, `Social_Person`

**Preprocessing Pipeline**:
1. Convert input dict to pandas DataFrame
2. Extract feature columns (categorical + numeric)
3. Apply ColumnTransformer:
   - Categorical → OneHotEncoder
   - Numeric → StandardScaler
4. Concatenate transformed features
5. Convert to float32 numpy array
6. Shape: `[1, num_features]` (single sample)

**Inference Pipeline**:
1. Get input name from ONNX session
2. Run session with preprocessed features
3. Get logits output (5 classes)
4. Apply softmax (numerically stable)
5. Get predicted class index (0-4)
6. Map to stress level (1-5):
   - If label encoder exists: `inverse_transform([idx])`
   - Otherwise: `idx + 1`
7. Map to stress label (Low, Medium, High, etc.)
8. Build response dictionary

**Stress Level Labels**:
- 1: "Low"
- 2: "Low-Medium"
- 3: "Medium"
- 4: "Medium-High"
- 5: "High"

---

### StressLevelInput (Pydantic Model)

**Purpose**: Input validation model for API requests.

**Key Fields** (all required):
- `Avg_Working_Hours_Per_Day`: float (0-24 range)
- `Work_From`: str (dropdown selection)
- `Work_Pressure`: int (1-5 range)
- `Manager_Support`: int (1-5 range)
- `Sleeping_Habit`: int (1-5 range)
- `Exercise_Habit`: int (1-5 range)
- `Job_Satisfaction`: int (1-5 range)
- `Work_Life_Balance`: str (Yes/No)
- `Social_Person`: int (1-5 range)
- `Lives_With_Family`: str (Yes/No)
- `Working_State`: str (dropdown selection)

**Validation**: Pydantic automatically validates field types and constraints (ge, le).

---

### StressLevelPrediction (Pydantic Model)

**Purpose**: Response model for API validation and serialization.

**Key Fields**:
- `stress_level`: Predicted stress level (1-5)
- `stress_label`: Human-readable label (Low, Medium, High, etc.)
- `confidence`: Confidence score (0-1) for prediction
- `probabilities`: Dictionary with probabilities for each class (class_1 to class_5)
- `all_probabilities`: List of dictionaries with detailed probabilities (level, label, probability)

**Validation**: Automatically validates response structure and types.

---

## Data Flow Summary

### Request Flow

1. **Client** → HTTP Request (form-data or JSON with 11 fields)
2. **FastAPI** → Route handler receives request
3. **Validator** → Validates input (Pydantic for API, form extraction for web)
4. **Predictor** → Converts to DataFrame and preprocesses
5. **ONNX Runtime** → Runs inference on model
6. **Predictor** → Postprocesses results (softmax, label mapping)
7. **Route** → Formats response (HTML or JSON)
8. **Client** ← HTTP Response with stress level prediction

### Model Loading Flow

1. **Application Startup** → Lifespan handler triggered
2. **Predictor Creation** → Initialize with model paths
3. **ONNX Loading** → Load model.onnx into InferenceSession
4. **Preprocessor Loading** → Load preprocessor.joblib (ColumnTransformer)
5. **Encoder Loading** → Load label_encoder.joblib (optional)
6. **Ready** → Predictor available for requests

### Preprocessing Flow

1. **Input Dict** → Convert to pandas DataFrame
2. **Feature Extraction** → Separate categorical and numeric columns
3. **One-Hot Encoding** → Transform 4 categorical columns
4. **Standardization** → Transform 7 numeric columns
5. **Concatenation** → Combine into single feature array
6. **Type Conversion** → Convert to float32
7. **ONNX Input** → Ready for inference

---

## Key Design Patterns

1. **Singleton Pattern**: Global predictor instance (loaded once at startup)
2. **Service Layer Pattern**: Predictor encapsulates inference logic
3. **Template Method Pattern**: FastAPI defines request/response structure
4. **Factory Pattern**: Preprocessor created from saved joblib file
5. **Adapter Pattern**: Predictor adapts ONNX model to application needs

---

## Integration Points

### ONNX Model Integration

- **Model Format**: ONNX (Open Neural Network Exchange)
- **Runtime**: ONNX Runtime (ort.InferenceSession)
- **Input Shape**: [1, num_features] (single sample, preprocessed features)
- **Output Shape**: [5] (logits for 5 stress levels)
- **Preprocessing**: Must match training preprocessing (ColumnTransformer)

### Preprocessor Integration

- **Format**: sklearn ColumnTransformer saved with joblib
- **Purpose**: One-hot encoding for categorical, standardization for numeric
- **Loading**: Loaded at startup, used for all predictions
- **Consistency**: Must match training-time preprocessing exactly

### Label Encoder Integration

- **Format**: sklearn LabelEncoder saved with joblib (optional)
- **Purpose**: Maps model output (0-indexed) to original labels (1-5)
- **Loading**: Loaded at startup if available
- **Fallback**: If not available, uses simple mapping (idx + 1)

### Web Interface Integration

- **Template Engine**: Jinja2
- **Static Files**: CSS (served via FastAPI static mount)
- **Form Inputs**: HTML form with sliders, dropdowns, number inputs
- **Dynamic Options**: Dropdown options loaded from dataset CSV
- **Interactive**: Real-time slider value display, form validation

---

## Comparison: Image vs Tabular API

| Aspect | Food Classification | Stress Level Prediction |
|--------|-------------------|------------------------|
| **Input Type** | Image files | Form data (tabular) |
| **Input Method** | File upload | Form fields |
| **Preprocessing** | Image transforms (resize, normalize) | sklearn (one-hot, standardization) |
| **Input Shape** | [1, 3, 224, 224] (image tensor) | [1, num_features] (feature vector) |
| **Output** | Food class names | Stress level (1-5) + label |
| **Model Type** | CNN | Feedforward NN |
| **Label Mapping** | Direct class names | Index to level (0-4 → 1-5) |

---

*Last Updated: [Current Date]*
