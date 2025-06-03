#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Central configuration file for BERT model training
Defines available models and training parameters to ensure consistency across different training modes
"""

# Available models
AVAILABLE_MODELS = {
    "Bio_ClinicalBERT": "emilyalsentzer/Bio_ClinicalBERT",
    "ClinicalBERT-AE-NER": "MutazYoune/ClinicalBERT-AE-NER",
    "ModernBERT-base-biomedical-ner": "Kushtrim/ModernBERT-base-biomedical-ner",
}

# Training modes and compatible models
TRAINING_MODES = {
    "standard": {
        "description": "Basic BERT model training with standard parameters",
        "compatible_models": list(AVAILABLE_MODELS.keys()),
        "module": "Step_2_train_BERT_models.train_bert_models",
        "function": "train_models",
    },
    "enhanced": {
        "description": "Enhanced BERT training with CRF layer, improved token alignment, and early stopping",
        "compatible_models": list(AVAILABLE_MODELS.keys()),
        "module": "Step_2_train_BERT_models.enhanced_train_bert_models",
        "function": "train_models",
    },
    "modernbert": {
        "description": "Specialized training for ModernBERT architecture without token_type_ids",
        "compatible_models": ["ModernBERT-base-biomedical-ner"],
        "module": "Step_2_train_BERT_models.enhanced_train_modernbert_models",
        "function": "train_modernbert_models",
    }
}

# Default training parameters
DEFAULT_PARAMETERS = {
    "standard": {
        "batch_size": 16,
        "epochs": 3,
        "learning_rate": 5e-5,
        "max_length": 128,
        "val_size": 0.1,
    },
    "enhanced": {
        "batch_size": 16,
        "epochs": 10,  # Increased epochs with early stopping
        "learning_rate": 3e-5,  # Slightly lower for stability
        "max_length": 128,
        "val_size": 0.1,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "gradient_clip": 1.0,
        "use_crf": True,
    },
    "modernbert": {
        "batch_size": 16,
        "epochs": 10,
        "learning_rate": 3e-5,
        "max_length": 128,
        "val_size": 0.1,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "gradient_clip": 1.0,
        "use_crf": True,
    }
}

# Data sources to use for training
DATA_SOURCES = ["direct", "dspy"]

# Default max sequence length for all models
DEFAULT_MAX_LENGTH = 128

def get_model_path(model_name):
    """Get the full model path for a given model name"""
    if model_name in AVAILABLE_MODELS:
        return AVAILABLE_MODELS[model_name]
    return model_name

def get_training_config(training_mode, **overrides):
    """Get the training configuration for a given mode with optional overrides"""
    if training_mode not in DEFAULT_PARAMETERS:
        raise ValueError(f"Unknown training mode: {training_mode}")
    
    # Start with default parameters for the mode
    config = DEFAULT_PARAMETERS[training_mode].copy()
    
    # Apply any overrides
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    
    return config

def validate_model_for_mode(model_name, training_mode):
    """Validate that the selected model is compatible with the training mode"""
    if training_mode not in TRAINING_MODES:
        raise ValueError(f"Unknown training mode: {training_mode}")
    
    compatible_models = TRAINING_MODES[training_mode]["compatible_models"]
    if model_name not in compatible_models:
        raise ValueError(f"Model {model_name} is not compatible with {training_mode} mode. Compatible models: {', '.join(compatible_models)}")
    
    return True 