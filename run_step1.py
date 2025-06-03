#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run Step 1: Data Generation
---------------------------
This script is the main entry point for running the data generation step
of the ADE extraction pipeline. It generates training data using configurable data generation approaches.
"""

import logging
import os
import importlib
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# CONFIGURATION
# List of supported data sources - add new sources here
AVAILABLE_DATA_SOURCES = ["direct", "dspy", "pipeline", "validator", "structured"]

# Which data sources to actually use (True = enabled, False = disabled)
# By default, only direct is enabled
ENABLED_SOURCES = {
    "direct": True,    # Direct LLM extraction (OpenAI API)
    "dspy": False,      # DSPy-based extraction framework
    "pipeline": False,  # Multi-step pipeline extraction approach
    "validator": False, # Simple extraction with validation for missed entities
    "structured": False # Structured DSPy-based extraction with position-aware entities
}

def main():
    """Main function to configure and run data generation."""
    # Base configuration
    base_config = {
        # Output options
        "output_dir": os.path.join("Step_1_data_generation", "data"),
        
        # Input file path
        "input_file": os.path.join("Step_1_data_generation", "data", "train.txt"),
        
        # Generation options
        "max_notes": 20,          # Maximum number of notes to process
        "overwrite_cache": True,  # Force regeneration (costs money)
        
        # LLM options
        "model_name": "gpt-4.1-nano-2025-04-14",
        "temperature": 0.1,
        "max_tokens": 2000,
        "batch_size": 10,          # Batch size for LLM processing
        
        # Dataset options
        "seed": 42                 # Random seed for reproducibility
    }
    
    # Filter to only use available sources
    active_sources = [source for source in AVAILABLE_DATA_SOURCES 
                     if ENABLED_SOURCES.get(source, False)]
    
    # Dynamically build the full config with paths for each data source
    config = base_config.copy()
    
    # Add file paths for each data source
    for source in AVAILABLE_DATA_SOURCES:
        # File paths specific to this data source
        config[f"{source}_output_file"] = os.path.join("Step_1_data_generation", "data", source, "ner_data.jsonl")
        config[f"{source}_extracted_file"] = os.path.join("Step_1_data_generation", "data", source, "extracted_data.jsonl")
    
    # Add gold data path
    config["gold_file"] = os.path.join("Step_1_data_generation", "data", "gold", "gold_ner_data.jsonl")
    
    # Results container
    results = {}
    
    if not active_sources:
        logger.warning("No data sources are enabled. Set at least one source to True in ENABLED_SOURCES.")
        return 0
    
    logger.info(f"Active data sources: {', '.join(active_sources)}")
    
    # Dynamically import and run each enabled data source's generator
    for source in active_sources:
        logger.info(f"Running {source}-based data generation...")
        
        try:
            # Dynamically import the generator module
            module_name = f"Step_1_data_generation.{source}_llm_generator"
            if source in ["dspy", "structured"]:
                module_name = f"Step_1_data_generation.{source}_generator"
                
            generator_module = importlib.import_module(module_name)
            
            # Get the generator function (follow naming convention: run_{source}_generation)
            generator_func = getattr(generator_module, f"run_{source}_generation")
            
            # Run the generator
            results[source] = generator_func(config)
            
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to run {source} generator: {str(e)}")
            results[source] = {"status": "error", "error": str(e)}
    
    logger.info("Data generation completed!")
    return 0

if __name__ == "__main__":
    exit(main()) 