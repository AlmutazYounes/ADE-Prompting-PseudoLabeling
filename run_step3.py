#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run Step 3: Model Evaluation
---------------------------
This script is the main entry point for running the model evaluation step
of the ADE extraction pipeline. It evaluates trained BERT models and compares
their performance to baseline approaches.
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
    """Main function to configure and run evaluation."""
    # Simple config dict for all options
    from Step_3_model_evaluation.config import AVAILABLE_DATA_SOURCES
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    config = {
        # Output options
        "output_dir": os.path.join("analysis", "comparison_results", f"{current_time}"),
        "name": None,  # Optional name to include in the output directory
        
        # Input paths
        "gold_data_path": os.path.join("Step_1_data_generation", "data", "gold", "gold_ner_data.jsonl"),
        "trained_models_dir": os.path.join("Step_2_train_BERT_models", "trained_models"),
        
        # Evaluation options
        "use_cache": True,           # Use cached results for LLM/DSPy evaluation
        "overwrite_cache": False,    # Force rerun of LLM and DSPy extraction (costs money)
        "skip_llm": False,           # Skip evaluation of LLM-based approaches
        "max_test_notes": 10,        # Number of test notes to evaluate
        "bert_max_length": 256,      # Max sequence length for BERT models
        
        # Additional analysis
        "detailed_analysis": False,   # Perform detailed error analysis (slower)

        # Data sources to evaluate (user can select from AVAILABLE_DATA_SOURCES)
        "data_sources": ["direct", "dspy", "pipeline", "validator", "structured"] 
    }

    # Import the consolidated evaluation module
    from Step_3_model_evaluation.evaluate import run_evaluation

    # Run the evaluation pipeline with the config dict
    return run_evaluation(config)

if __name__ == "__main__":
    exit(main()) 