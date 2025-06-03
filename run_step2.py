#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run Step 2: Train BERT Models
---------------------------
This script is the main entry point for running the BERT model training step
of the ADE extraction pipeline. It trains BERT models on the data generated in Step 1.
"""

import logging
import os
from datetime import datetime

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
        
        # Training options
        "models_to_train": ["MutazYoune_ClinicalBERT"],  # Model keys from AVAILABLE_MODELS dict
        "overwrite_existing": True,  # Overwrite existing trained models
        
        # Training parameters
        "batch_size": 16,
        "learning_rate": 5e-5,
        "epochs": 3,
        "max_length": 128,  # Match DEFAULT_MAX_LENGTH in bert_config.py
        "val_size": 0.1,    # Validation split size
        "seed": 42,         # Random seed for reproducibility
        
        # Enhanced training flag
        "use_enhanced_training": True,  # Set to True to use the enhanced training approach
    }

    # Import the BERT model training module
    if config["use_enhanced_training"]:
        from Step_2_train_BERT_models.enhanced_train_bert_models import train_models
        logger.info("Using enhanced BERT model training...")
    else:
        from Step_2_train_BERT_models.train_bert_models import train_models
        logger.info("Using standard BERT model training...")

    # Run the model training pipeline with the config dict
    logger.info("Starting BERT model training...")
    train_models(config)
    logger.info("BERT model training completed!")
    return 0

if __name__ == "__main__":
    exit(main()) 