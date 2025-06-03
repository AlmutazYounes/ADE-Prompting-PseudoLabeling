#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run Step 2: Train BERT Models
---------------------------
This script is the main entry point for running the BERT model training step
of the ADE extraction pipeline. It trains BERT models on the data generated in Step 1.

Training Modes:
- standard: Basic training for all BERT-based models
- enhanced: Enhanced training with CRF and optimizations (two implementations):
  * Regular BERT models: Uses enhanced_train_bert_models.py
  * ModernBERT: Uses specialized enhanced_train_modernbert_models.py
"""

import logging
import os
import importlib
from datetime import datetime
from Step_2_train_BERT_models.bert_config import (
    AVAILABLE_MODELS, 
    TRAINING_MODES, 
    DATA_SOURCES, 
    get_training_config,
    validate_model_for_mode
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to configure and run BERT model training."""
    # Simple config dict for all options
    config = {
        # Output options
        "output_dir": os.path.join("Step_2_train_BERT_models", "trained_models"),
        
        # Data sources and paths
        "data_sources": ["direct"],  # Which data sources to use for training
        "dataset_filename": "ner_data_4.1-nano_enhanced_direct_approach.jsonl",  # Dataset file to use (options: ner_data.jsonl, ner_data_41nano.jsonl, ner_data_4.1-nano_enhanced_direct_approach.jsonl)
        
        # Training options
        # Available models: Bio_ClinicalBERT, ClinicalBERT-AE-NER, ModernBERT-base-biomedical-ner
        "models_to_train": ["ModernBERT-base-biomedical-ner"],  
        "overwrite_existing": True,  # Overwrite existing trained models
        
        # Training parameters
        "batch_size": 16,
        "learning_rate": 5e-5,
        "epochs": 3,
        "max_length": 256,
        "val_size": 0.1,
        "seed": 42,
        
        # Training mode flag
        # Options:
        # - "standard": Basic training for all models
        # - "enhanced": Enhanced training for BERT models (CRF layer, early stopping, etc.)
        # - "modernbert": Specialized enhanced training for ModernBERT models
        "training_mode": "modernbert",
    }

    # Validate model and training mode compatibility
    for model in config["models_to_train"]:
        try:
            validate_model_for_mode(model, config["training_mode"])
        except ValueError as e:
            logger.error(str(e))
            return 1

    # Get training mode configuration
    training_mode = config["training_mode"]
    if training_mode not in TRAINING_MODES:
        logger.error(f"Invalid training mode: {training_mode}")
        logger.info(f"Available modes: {', '.join(TRAINING_MODES.keys())}")
        return 1

    # Get the appropriate training function based on training mode
    mode_config = TRAINING_MODES[training_mode]
    logger.info(f"Using {training_mode} training mode: {mode_config['description']}")
    
    # Import the appropriate training module dynamically
    module_path = mode_config["module"]
    function_name = mode_config["function"]
    
    try:
        module = importlib.import_module(module_path)
        train_function = getattr(module, function_name)
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to import training function: {e}")
        return 1

    # Merge configuration with mode defaults
    training_config = get_training_config(training_mode, **{
        k: config[k] for k in [
            "batch_size", "learning_rate", "epochs", 
            "max_length", "val_size", "seed"
        ] if k in config
    })
    
    # Add non-training parameters
    for key in ["output_dir", "data_sources", "dataset_filename", "models_to_train", "overwrite_existing"]:
        if key in config:
            training_config[key] = config[key]

    # Run the model training pipeline with the config dict
    logger.info("Starting BERT model training...")
    logger.info(f"Training {', '.join(config['models_to_train'])} with {training_mode} mode")
    logger.info(f"Using dataset: {config['dataset_filename']}")
    train_function(training_config)
    logger.info("BERT model training completed!")
    return 0

if __name__ == "__main__":
    exit(main()) 