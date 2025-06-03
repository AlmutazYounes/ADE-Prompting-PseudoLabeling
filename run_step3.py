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

def select_llm_approaches():
    """
    Allow user to interactively select which LLM approaches to evaluate.
    
    Returns:
        list: Selected LLM approach names
    """
    available_approaches = {
        "1": "direct",   # Direct LLM
        "2": "dspy",     # DSPy
        "3": "pipeline", # Pipeline LLM
        "4": "validator", # Validator LLM
        "5": "structured" # Structured DSPy
    }
    
    print("\n" + "="*60)
    print("SELECT LLM APPROACHES TO EVALUATE")
    print("="*60)
    print("Available LLM approaches:")
    print("  1. Direct LLM (simple one-shot extraction)")
    print("  2. DSPy (extraction with DSPy framework)")
    print("  3. Pipeline LLM (multi-step extraction pipeline)")
    print("  4. Validator LLM (two-step extraction with validation)")
    print("  5. Structured DSPy (structured entity extraction with DSPy)")
    print("\nEnter approach numbers separated by commas (e.g., '1,3,5')")
    print("Or enter 'all' to evaluate all approaches")
    print("-"*60)
    
    while True:
        selection = input("Select approaches to evaluate: ").strip().lower()
        
        if selection == 'all':
            return list(available_approaches.values())
        
        try:
            selected_numbers = [s.strip() for s in selection.split(',')]
            selected_approaches = []
            
            for num in selected_numbers:
                if num in available_approaches:
                    selected_approaches.append(available_approaches[num])
                else:
                    print(f"Invalid selection: {num}. Please use numbers 1-5.")
                    break
            else:
                if selected_approaches:
                    print(f"\nSelected approaches: {', '.join(selected_approaches)}")
                    return selected_approaches
                else:
                    print("Please select at least one approach.")
        except Exception as e:
            print(f"Error: {str(e)}. Please try again.")

def configure_evaluation_settings():
    """
    Allow user to configure evaluation settings like number of test notes and caching.
    
    Returns:
        dict: Dictionary with evaluation settings
    """
    settings = {}
    
    print("\n" + "="*60)
    print("CONFIGURE EVALUATION SETTINGS")
    print("="*60)
    
    # Optional run name
    run_name = input("Name for this evaluation run (optional): ").strip()
    settings['name'] = run_name if run_name else None
    
    # Configure number of test notes
    while True:
        try:
            test_notes = input("Number of test notes to evaluate (10-1000, default=100): ").strip()
            if not test_notes:
                settings['max_notes'] = 100
                break
                
            test_notes = int(test_notes)
            if 10 <= test_notes <= 1000:
                settings['max_notes'] = test_notes
                break
            else:
                print("Please enter a number between 10 and 1000.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Configure caching behavior
    use_cache = input("Use cached results if available? (y/n, default=y): ").strip().lower()
    settings['use_cache'] = use_cache != 'n'
    
    if settings['use_cache']:
        overwrite = input("Force regeneration of cached results? (y/n, default=n): ").strip().lower()
        settings['overwrite_cache'] = overwrite == 'y'
    else:
        settings['overwrite_cache'] = True  # If not using cache, must regenerate
    
    print("\nEvaluation Settings:")
    if settings['name']:
        print(f"  - Run Name: {settings['name']}")
    print(f"  - Test Notes: {settings['max_notes']}")
    print(f"  - Use Cache: {settings['use_cache']}")
    print(f"  - Overwrite Cache: {settings['overwrite_cache']}")
    
    return settings

def main():
    """Main function to configure and run evaluation."""
    # Simple config dict for all options
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prompt user for LLM approaches and evaluation settings
    selected_approaches = select_llm_approaches()
    settings = configure_evaluation_settings()
    
    # Create directory name with optional custom name
    if settings.get('name'):
        dir_name = f"{current_time}_{settings['name']}"
    else:
        dir_name = current_time
    
    config = {
        # Output options
        "output_dir": os.path.join("analysis", "comparison_results", dir_name),
        "name": settings.get('name'),  # Optional name to include in the output directory
        
        # Input paths
        "gold_data_path": os.path.join("Step_1_data_generation", "data", "gold", "gold_ner_data.jsonl"),
        "trained_models_dir": os.path.join("Step_2_train_BERT_models", "trained_models"),
        
        # Evaluation options
        "use_cache": settings['use_cache'],           # Use cached results for LLM/DSPy evaluation
        "overwrite_cache": settings['overwrite_cache'],    # Force rerun of LLM and DSPy extraction (costs money)
        "skip_llm": False,           # Skip evaluation of LLM-based approaches
        "max_notes": settings['max_notes'],        # Number of test notes to evaluate
        "bert_max_length": 256,      # Max sequence length for BERT models
        
        # Additional analysis
        "detailed_analysis": False,   # Perform detailed error analysis (slower)

        # Data sources to evaluate (user can select from AVAILABLE_DATA_SOURCES)
        # Available data sources: direct, dspy, pipeline, validator, structured
        "data_sources": selected_approaches 
    }

    # Import the consolidated evaluation module
    from Step_3_model_evaluation import run_evaluation

    # Run the evaluation pipeline with the config dict
    return run_evaluation(config)

if __name__ == "__main__":
    exit(main()) 