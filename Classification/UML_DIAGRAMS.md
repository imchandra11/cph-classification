# Classification Module - UML Diagrams & Architecture Documentation

This document provides comprehensive UML diagrams and architectural documentation for the Classification module, a reusable PyTorch Lightning pipeline for tabular classification tasks.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Class Diagrams](#class-diagrams)
3. [Sequence Diagrams](#sequence-diagrams)
4. [Component Diagrams](#component-diagrams)
5. [Activity Diagrams](#activity-diagrams)
6. [Class-Level Details](#class-level-details)

---

## Architecture Overview

The Classification module is a generic, config-driven pipeline for training classification models on tabular data. It extends the Regression module's architecture with classification-specific features:

- **Multi-class Classification**: Supports any number of classes
- **Classification Metrics**: Accuracy, F1-Score, Precision, Recall
- **Class Weights**: Handles imbalanced datasets
- **Label Encoding**: Auto-encodes categorical labels to 0-indexed integers
- **Probability Outputs**: Softmax for class probabilities

Key differences from Regression:
- Output dimension = number of classes (not 1)
- Loss function = CrossEntropyLoss (not MSELoss)
- Metrics = Classification metrics (not regression metrics)
- Label preprocessing = Automatic encoding to ensure 0-indexed labels

---

## Class Diagrams

### Complete Class Diagram

```mermaid
classDiagram
    class CLSLightningCLI {
        -config: LightningConfig
        -model: ModelModuleCLS
        -datamodule: DataModuleCLS
        -trainer: Trainer
        +add_arguments_to_parser(parser)
        +before_instantiate_classes()
        +after_instantiate_classes()
        +link_input_dim_and_num_classes()
    }
    
    class DataModuleCLS {
        -csv_path: Path
        -batch_size: int
        -num_workers: int
        -val_split: float
        -test_split: Optional[float]
        -random_seed: int
        -categorical_cols: list[str]
        -numeric_cols: list[str]
        -target_col: str
        -preprocessor: ColumnTransformer
        -label_encoder: Optional[LabelEncoder]
        -input_dim: int
        -num_classes: int
        -class_names: list[str]
        -train_df: DataFrame
        -val_df: DataFrame
        -test_df: DataFrame
        +setup(stage: str)
        +train_dataloader() DataLoader
        +val_dataloader() DataLoader
        +test_dataloader() DataLoader
        +get_input_dim() int
        +get_num_classes() int
        -_create_preprocessor()
        -_encode_labels()
        -_save_preprocessor()
    }
    
    class ClassificationDataset {
        -data: DataFrame
        -preprocessor: ColumnTransformer
        -target_col: str
        -feature_cols: list[str]
        -features: DataFrame
        -targets: ndarray
        -transformed_features: ndarray
        -features_tensor: Tensor
        -targets_tensor: Tensor
        +__len__() int
        +__getitem__(idx) tuple[Tensor, Tensor]
    }
    
    class ModelModuleCLS {
        -model: ClassificationModel
        -criterion: CrossEntropyLoss
        -lr: float
        -weight_decay: float
        -lr_scheduler_factor: Optional[float]
        -lr_scheduler_patience: Optional[int]
        -class_weights: Optional[Tensor]
        -num_classes: int
        -save_dir: Optional[str]
        -name: Optional[str]
        -train_accuracy: MulticlassAccuracy
        -val_accuracy: MulticlassAccuracy
        -test_accuracy: MulticlassAccuracy
        -train_f1: MulticlassF1Score
        -val_f1: MulticlassF1Score
        -test_f1: MulticlassF1Score
        -train_precision: MulticlassPrecision
        -val_precision: MulticlassPrecision
        -test_precision: MulticlassPrecision
        -train_recall: MulticlassRecall
        -val_recall: MulticlassRecall
        -test_recall: MulticlassRecall
        -training_step_outputs: list
        -validation_step_outputs: list
        +forward(x: Tensor) Tensor
        +training_step(batch, batch_idx) Tensor
        +validation_step(batch, batch_idx) Tensor
        +test_step(batch, batch_idx) dict
        +on_training_epoch_end()
        +on_validation_epoch_end()
        +configure_optimizers() Optimizer
        +configure_schedulers() LRScheduler
    }
    
    class ClassificationModel {
        -input_dim: int
        -num_classes: int
        -hidden_layers: list[int]
        -dropout_rates: list[float]
        -activation: str
        -model: nn.Sequential
        +forward(x: Tensor) Tensor
        +get_input_dim() int
        +get_num_classes() int
        +set_input_dim(dim: int)
        +set_num_classes(classes: int)
        -_get_activation(name: str) nn.Module
        -_build_model()
    }
    
    class ONNXExportCallback {
        -output_dir: Path
        -model_name: str
        -input_dim: Optional[int]
        +on_train_end(trainer, pl_module)
        -_determine_input_dim(trainer, pl_module) int
        -_export_to_onnx(model, input_dim)
    }
    
    %% Relationships
    CLSLightningCLI --> DataModuleCLS : creates & configures
    CLSLightningCLI --> ModelModuleCLS : creates & configures
    CLSLightningCLI --> Trainer : creates
    
    DataModuleCLS --> ClassificationDataset : creates instances
    DataModuleCLS ..> sklearn.preprocessing : uses StandardScaler, OneHotEncoder, LabelEncoder
    DataModuleCLS ..> sklearn.compose : uses ColumnTransformer
    DataModuleCLS ..> joblib : saves/loads preprocessor & label_encoder
    
    ClassificationDataset --> torch.utils.data.Dataset : extends
    ClassificationDataset --> pandas : uses DataFrame
    ClassificationDataset --> numpy : uses arrays
    
    ModelModuleCLS --> ClassificationModel : contains
    ModelModuleCLS --> lightning.LightningModule : extends
    ModelModuleCLS --> torch.nn : uses CrossEntropyLoss
    ModelModuleCLS --> torchmetrics : uses Accuracy, F1Score, Precision, Recall
    
    ClassificationModel --> torch.nn.Module : extends
    ClassificationModel --> torch.nn : uses Linear, Dropout, etc.
    
    Trainer --> ONNXExportCallback : uses via callbacks
    Trainer --> ModelModuleCLS : trains
    Trainer --> DataModuleCLS : uses for data
```

### Key Differences from Regression

```mermaid
classDiagram
    class BaseDataModule {
        <<abstract>>
        +setup()
        +get_input_dim()
    }
    
    class BaseModelModule {
        <<abstract>>
        +training_step()
        +configure_optimizers()
    }
    
    class BaseModel {
        <<abstract>>
        +forward()
    }
    
    DataModuleRGS --|> BaseDataModule
    DataModuleCLS --|> BaseDataModule
    ModelModuleRGS --|> BaseModelModule
    ModelModuleCLS --|> BaseModelModule
    RegressionModel --|> BaseModel
    ClassificationModel --|> BaseModel
    
    DataModuleCLS : +get_num_classes()
    DataModuleCLS : +label_encoder
    DataModuleCLS : +class_names
    
    ModelModuleCLS : +CrossEntropyLoss
    ModelModuleCLS : +class_weights
    ModelModuleCLS : +accuracy_metrics
    ModelModuleCLS : +f1_metrics
    
    ClassificationModel : +num_classes
    ClassificationModel : +output_dim = num_classes
```

---

## Sequence Diagrams

### Training Workflow Sequence

```mermaid
sequenceDiagram
    participant User
    participant CLI as CLSLightningCLI
    participant Config as config.yaml
    participant DM as DataModuleCLS
    participant Dataset as ClassificationDataset
    participant ModelModule as ModelModuleCLS
    participant NNModel as ClassificationModel
    participant Trainer
    participant Callback as ONNXExportCallback
    participant Files as File System

    User->>CLI: python main.py --config config.yaml
    CLI->>Config: Load configuration
    Config-->>CLI: Configuration dict
    
    Note over CLI: Phase 1: Configuration & Setup
    CLI->>CLI: before_instantiate_classes()
    CLI->>DM: Create DataModuleCLS(csv_path, ...)
    CLI->>DM: setup('fit')
    DM->>Files: Read CSV file
    Files-->>DM: DataFrame
    
    DM->>DM: Extract target column
    DM->>DM: Check if labels are 0-indexed
    DM->>DM: _encode_labels() [if needed]
    DM->>DM: Split train/val/test
    DM->>DM: _create_preprocessor()
    DM->>DM: Fit preprocessor on train data
    DM->>DM: Calculate input_dim
    DM->>DM: Calculate num_classes
    DM->>Files: Save preprocessor.joblib
    DM->>Files: Save label_encoder.joblib
    DM-->>CLI: input_dim, num_classes calculated
    
    CLI->>CLI: Auto-link input_dim and num_classes to model
    CLI->>NNModel: Create ClassificationModel(input_dim, num_classes, ...)
    NNModel->>NNModel: _build_model()
    CLI->>ModelModule: Create ModelModuleCLS(model, ...)
    ModelModule->>ModelModule: Initialize metrics (Accuracy, F1, Precision, Recall)
    CLI->>Trainer: Create Trainer(callbacks=[...])
    CLI->>Callback: Register ONNXExportCallback
    
    Note over CLI: Phase 2: Training Loop
    CLI->>Trainer: fit(model, datamodule)
    
    loop For each epoch
        Trainer->>DM: train_dataloader()
        DM->>Dataset: Create ClassificationDataset(train_df)
        Dataset->>Dataset: Transform features
        Dataset-->>DM: DataLoader(train)
        DM-->>Trainer: DataLoader
        
        loop For each batch
            Trainer->>ModelModule: training_step(batch, batch_idx)
            ModelModule->>NNModel: forward(features)
            NNModel->>NNModel: Forward through layers
            NNModel-->>ModelModule: logits [batch_size, num_classes]
            ModelModule->>ModelModule: criterion(logits, targets)
            ModelModule->>ModelModule: Calculate CrossEntropy loss
            ModelModule->>ModelModule: Update accuracy metrics
            ModelModule->>ModelModule: Update F1, precision, recall
            ModelModule-->>Trainer: loss
            Trainer->>ModelModule: backward()
            Trainer->>ModelModule: optimizer_step()
        end
        
        Trainer->>DM: val_dataloader()
        DM-->>Trainer: DataLoader(val)
        
        loop For each validation batch
            Trainer->>ModelModule: validation_step(batch, batch_idx)
            ModelModule->>ModelModule: Calculate val metrics
            ModelModule->>ModelModule: Update val_accuracy, val_f1, etc.
            ModelModule-->>Trainer: val_loss, val_accuracy, val_f1
        end
        
        ModelModule->>ModelModule: on_training_epoch_end()
        ModelModule->>ModelModule: on_validation_epoch_end()
        ModelModule->>ModelModule: Log aggregated metrics
    end
    
    Note over CLI: Phase 3: Model Export
    Trainer->>Callback: on_train_end(trainer, pl_module)
    Callback->>DM: get_input_dim()
    DM-->>Callback: input_dim
    Callback->>NNModel: Export to ONNX
    Callback->>Files: Save model.onnx
    Callback-->>User: ✓ Model exported
```

### Label Encoding Sequence

```mermaid
sequenceDiagram
    participant DM as DataModuleCLS
    participant DF as DataFrame
    participant LabelEnc as LabelEncoder
    participant Store as Storage

    DM->>DF: Extract target column
    DF-->>DM: Raw labels [e.g., 1,2,3,4,5 or "A","B","C"]
    
    DM->>DM: Check if labels are integers
    alt Labels are integers
        DM->>DM: Check if 0-indexed
        alt Not 0-indexed [e.g., 1-5]
            DM->>LabelEnc: Create LabelEncoder
            DM->>LabelEnc: fit_transform(labels)
            LabelEnc-->>DM: 0-indexed labels [0,1,2,3,4]
            DM->>DM: Update class_names from encoder
        else Already 0-indexed
            DM->>DM: Use labels as-is
            DM->>DM: Set class_names from unique labels
        end
    else Labels are strings
        DM->>LabelEnc: Create LabelEncoder
        DM->>LabelEnc: fit_transform(labels)
        LabelEnc-->>DM: Encoded labels [0,1,2,...]
        DM->>DM: Store class_names from encoder
    end
    
    DM->>DM: Update DataFrame with encoded labels
    DM->>Store: Save label_encoder.joblib
    DM->>DM: num_classes = len(unique_labels)
    DM-->>DM: Return num_classes
```

---

## Component Diagrams

### System Architecture

```mermaid
graph TB
    subgraph "Entry Points"
        Main1[main.py<br/>Standard CLI]
        Main2[mainfittest.py<br/>Fit+Test Workflow]
    end
    
    subgraph "CLI Layer"
        CLI[CLSLightningCLI<br/>Custom Lightning CLI]
        ConfigParser[Config Parser<br/>YAML Handler]
    end
    
    subgraph "Configuration"
        YAML[config.yaml<br/>Hyperparameters<br/>Paths<br/>Model Config]
    end
    
    subgraph "Data Layer"
        DM[DataModuleCLS<br/>Lightning DataModule]
        Dataset[ClassificationDataset<br/>PyTorch Dataset]
        Preprocessor[ColumnTransformer<br/>sklearn Pipeline]
        LabelEncoder[LabelEncoder<br/>Label Encoding]
        CSV[CSV Data File]
        PreprocessorFile[preprocessor.joblib]
        LabelEncoderFile[label_encoder.joblib]
    end
    
    subgraph "Model Layer"
        ModelModule[ModelModuleCLS<br/>Lightning Module]
        NNModel[ClassificationModel<br/>Feedforward NN]
        Optimizer[Optimizer<br/>Adam/SGD/RMSprop]
        Scheduler[LR Scheduler<br/>ReduceLROnPlateau]
        ClassWeights[Class Weights<br/>For Imbalanced Data]
    end
    
    subgraph "Training Layer"
        Trainer[PyTorch Lightning Trainer]
        Metrics[Classification Metrics<br/>Accuracy, F1, Precision, Recall]
        Logger[Logger<br/>TensorBoard/CSV]
        Checkpoint[Model Checkpoint<br/>Best/Last]
    end
    
    subgraph "Export Layer"
        Callback[ONNXExportCallback]
        ONNX[model.onnx<br/>Production Model]
    end
    
    subgraph "External Libraries"
        Lightning[PyTorch Lightning]
        PyTorch[PyTorch]
        sklearn[scikit-learn]
        torchmetrics[TorchMetrics]
    end
    
    %% Connections
    Main1 --> CLI
    Main2 --> CLI
    CLI --> ConfigParser
    YAML --> ConfigParser
    ConfigParser --> CLI
    
    CLI --> DM
    CLI --> ModelModule
    CLI --> Trainer
    
    DM --> CSV
    DM --> Preprocessor
    DM --> LabelEncoder
    DM --> Dataset
    DM --> PreprocessorFile
    DM --> LabelEncoderFile
    
    Preprocessor --> sklearn
    LabelEncoder --> sklearn
    
    Dataset --> PyTorch
    
    ModelModule --> NNModel
    ModelModule --> Optimizer
    ModelModule --> Scheduler
    ModelModule --> ClassWeights
    ModelModule --> Lightning
    
    NNModel --> PyTorch
    
    Trainer --> ModelModule
    Trainer --> DM
    Trainer --> Metrics
    Trainer --> Logger
    Trainer --> Checkpoint
    Trainer --> Callback
    
    Metrics --> torchmetrics
    
    Callback --> ONNX
    
    style CLI fill:#667eea,color:#fff
    style ModelModule fill:#764ba2,color:#fff
    style Trainer fill:#f093fb,color:#fff
    style LabelEncoder fill:#ffd700,color:#000
```

---

## Activity Diagrams

### Classification Training Process

```mermaid
flowchart TD
    Start([Start Training]) --> LoadConfig[Load config.yaml]
    LoadConfig --> ParseConfig[Parse Configuration]
    ParseConfig --> CreateDM[Create DataModuleCLS]
    CreateDM --> SetupDM[Setup DataModule]
    SetupDM --> LoadCSV[Load CSV Data]
    LoadCSV --> ExtractTargets[Extract Target Column]
    ExtractTargets --> CheckLabels{Label Type?}
    
    CheckLabels -->|Integers| CheckIndexed{0-indexed?}
    CheckLabels -->|Strings| EncodeStrings[Encode with LabelEncoder]
    
    CheckIndexed -->|Yes| UseAsIs[Use labels as-is]
    CheckIndexed -->|No| RemapIntegers[Remap to 0-indexed]
    
    EncodeStrings --> GetClasses
    UseAsIs --> GetClasses
    RemapIntegers --> GetClasses
    
    GetClasses[Get num_classes & class_names] --> SplitData{Split Data?}
    SplitData -->|Yes| TrainValTest[Split: Train/Val/Test]
    SplitData -->|No| TrainVal[Split: Train/Val]
    
    TrainValTest --> CreatePreprocessor[Create Preprocessor]
    TrainVal --> CreatePreprocessor
    CreatePreprocessor --> FitPreprocessor[Fit Preprocessor on Train]
    FitPreprocessor --> CalcInputDim[Calculate input_dim]
    CalcInputDim --> SaveArtifacts{Save Artifacts?}
    SaveArtifacts -->|Yes| SavePrep[Save preprocessor.joblib<br/>Save label_encoder.joblib]
    SaveArtifacts -->|No| CreateModel
    SavePrep --> CreateModel[Create ClassificationModel]
    
    CreateModel --> LinkDims[Auto-link input_dim & num_classes]
    LinkDims --> CreateModule[Create ModelModuleCLS]
    CreateModule --> InitMetrics[Initialize Metrics<br/>Accuracy, F1, Precision, Recall]
    InitMetrics --> CreateTrainer[Create Trainer]
    CreateTrainer --> AddCallbacks[Add Callbacks]
    
    AddCallbacks --> StartTraining[Start Training Loop]
    StartTraining --> Epoch{More Epochs?}
    
    Epoch -->|Yes| TrainBatch[Process Training Batch]
    TrainBatch --> ForwardPass[Forward Pass]
    ForwardPass --> CalcLoss[Calculate CrossEntropy Loss]
    CalcLoss --> UpdateMetrics[Update Classification Metrics]
    UpdateMetrics --> Backward[Backward Pass]
    Backward --> UpdateWeights[Update Weights]
    UpdateWeights --> ValBatch{Validation?}
    
    ValBatch -->|Yes| ValForward[Validation Forward]
    ValForward --> ValMetrics[Calculate Val Metrics<br/>Accuracy, F1, Precision, Recall]
    ValMetrics --> LogMetrics[Log Metrics]
    LogMetrics --> Checkpoint{Save Checkpoint?}
    
    ValBatch -->|No| Checkpoint
    Checkpoint -->|Yes| SaveCheckpoint[Save Model]
    Checkpoint -->|No| Epoch
    SaveCheckpoint --> Epoch
    
    Epoch -->|No| ExportONNX[Export to ONNX]
    ExportONNX --> End([Training Complete])
    
    style Start fill:#90EE90
    style End fill:#FFB6C1
    style ExportONNX fill:#87CEEB
    style EncodeStrings fill:#FFD700
    style RemapIntegers fill:#FFD700
```

---

## Class-Level Details

### CLSLightningCLI

**Purpose**: Custom CLI extending LightningCLI with classification-specific auto-linking.

**Key Responsibilities**:
- Parse YAML configuration
- Instantiate DataModule, Model, and Trainer
- Auto-link both `input_dim` and `num_classes` from DataModule to Model
- Handle checkpoint paths

**Key Methods**:
- `before_instantiate_classes()`: Auto-detects `input_dim` and `num_classes`
- `after_instantiate_classes()`: Fallback to set dimensions if needed

**Difference from Regression**: Also links `num_classes` in addition to `input_dim`.

---

### DataModuleCLS

**Purpose**: Manages data loading, preprocessing, and label encoding for classification.

**Key Responsibilities**:
- Load CSV data
- **Encode labels** to ensure 0-indexed integers (critical for CrossEntropyLoss)
- Create and fit preprocessing pipeline
- Split data into train/val/test
- Calculate input dimensions and number of classes
- Save preprocessor and label encoder

**Key Attributes**:
- `label_encoder`: sklearn LabelEncoder (for string labels or remapping)
- `num_classes`: Number of unique classes
- `class_names`: List of class names (for mapping predictions back)

**Key Methods**:
- `_encode_labels()`: Ensures labels are 0-indexed integers
- `get_num_classes()`: Returns number of classes
- `get_input_dim()`: Returns calculated input dimension

**Critical Feature**: Automatically remaps 1-indexed integer labels (e.g., 1-5) to 0-indexed (0-4) to match CrossEntropyLoss requirements.

---

### ClassificationDataset

**Purpose**: PyTorch Dataset for classification data with encoded labels.

**Key Difference from RegressionDataset**: Targets are LongTensor (integers) not FloatTensor (floats).

---

### ModelModuleCLS

**Purpose**: PyTorch Lightning module for classification with multi-class metrics.

**Key Responsibilities**:
- Define training/validation/test steps
- Use CrossEntropyLoss (not MSELoss)
- Track classification metrics: Accuracy, F1, Precision, Recall
- Support class weights for imbalanced data

**Key Attributes**:
- `criterion`: CrossEntropyLoss (with optional class weights)
- `train_accuracy`, `val_accuracy`, `test_accuracy`: MulticlassAccuracy metrics
- `train_f1`, `val_f1`, `test_f1`: MulticlassF1Score metrics
- Similar for Precision and Recall

**Key Methods**:
- `training_step()`: Returns CrossEntropy loss
- `validation_step()`: Updates accuracy, F1, precision, recall
- `on_training_epoch_end()`: Aggregates and logs metrics

---

### ClassificationModel

**Purpose**: Feedforward neural network for classification.

**Key Difference from RegressionModel**:
- `output_dim` = `num_classes` (not 1)
- Output layer size matches number of classes
- Forward pass produces logits for each class

**Key Methods**:
- `get_num_classes()`: Returns number of classes
- `set_num_classes()`: Sets number of classes if not known at init

---

## Key Design Patterns

1. **Template Method Pattern**: Lightning framework defines training loop
2. **Factory Pattern**: ModelFactory creates model instances
3. **Strategy Pattern**: Configurable optimizers, schedulers, activations
4. **Adapter Pattern**: LabelEncoder adapts labels to 0-indexed format
5. **Observer Pattern**: Callbacks observe training events

---

## Critical Implementation Details

### Label Encoding Strategy

The module ensures labels are always 0-indexed for CrossEntropyLoss:

1. **Integer Labels (e.g., 1-5)**:
   - Check if min label = 0
   - If not, remap: `encoded = original - min(original)`
   - Update class_names to reflect remapping

2. **String Labels (e.g., "cat", "dog")**:
   - Use LabelEncoder to map to 0, 1, 2, ...
   - Store mapping for inverse transform

3. **Already 0-indexed**:
   - Use as-is
   - Extract class_names from unique values

### Metrics Calculation

- **Accuracy**: Percentage of correct predictions
- **F1-Score**: Harmonic mean of precision and recall (macro-averaged)
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)

All metrics use macro-averaging (average across classes).

---

*Last Updated: [Current Date]*
