#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3: LLM Approach Evaluation
---------------------------
This script evaluates the LLM-based extraction approaches from Step 1
against the gold standard data.

The script produces performance metrics for LLM approaches that are saved to
the analysis folder.
"""

import os
import json
import shutil
import logging
from datetime import datetime
import numpy as np
from tqdm import tqdm

# Constants
STEP_2_GOLD_NER_DATA = os.path.join("Step_1_data_generation", "data", "gold", "gold_ner_data.jsonl")
LLM_CACHE_DIR = os.path.join("analysis", "llm_cache")
AVAILABLE_DATA_SOURCES = ["direct", "dspy", "pipeline", "validator", "structured"]
USE_CACHE_DEFAULT = True
OVERWRITE_CACHE_DEFAULT = False

#############################
# Utility Functions
#############################

def load_from_jsonl(file_path):
    """
    Load a list of JSON objects from a .jsonl file.
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    except Exception as e:
        print(f"❌ Failed to load data from {file_path}: {e}")
        return []

def save_to_jsonl(data, file_path):
    """
    Save a list of JSON objects to a .jsonl file.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def print_banner(text):
    """Print a nicely formatted banner with the given text."""
    term_width = shutil.get_terminal_size((80, 20)).columns
    banner_text = f" {text} "
    banner = f"\033[1;44m{banner_text.center(term_width)}\033[0m"
    print("\n" + banner + "\n")

#############################
# Evaluation Functions
#############################

def entity_level_metrics(pred_entities, gold_entities):
    """
    Compute entity-level precision, recall, and F1 for a single example.
    Entities are lists of dicts with 'start', 'end', and 'label'.
    """
    pred_set = {(e['start'], e['end'], e['label']) for e in pred_entities}
    gold_set = {(e['start'], e['end'], e['label']) for e in gold_entities}
    
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}

def calculate_overall_metrics(metrics_list):
    """
    Calculate overall metrics from a list of individual example metrics.
    
    Args:
        metrics_list: List of metric dictionaries
        
    Returns:
        Dictionary with overall metrics
    """
    if not metrics_list:
        return {"precision": 0, "recall": 0, "f1": 0}
    
    # Check if metrics are in the expected format
    if "overall" in metrics_list[0]:
        # Extract overall metrics from each example
        overall_metrics = [m.get("overall", {}) for m in metrics_list]
        
        # Calculate averages
        precision = np.mean([m.get("precision", 0) for m in overall_metrics])
        recall = np.mean([m.get("recall", 0) for m in overall_metrics])
        f1 = np.mean([m.get("f1", 0) for m in overall_metrics])
    else:
        # Assume flat structure
        precision = np.mean([m.get("precision", 0) for m in metrics_list])
        recall = np.mean([m.get("recall", 0) for m in metrics_list])
        f1 = np.mean([m.get("f1", 0) for m in metrics_list])
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }

def evaluate_llm_approaches(test_notes, gold_data, use_cache=USE_CACHE_DEFAULT, overwrite_cache=OVERWRITE_CACHE_DEFAULT, data_sources=None):
    """
    Evaluate LLM-based approaches from Step 1.
    Uses the actual implementations from Step 1 when cache needs to be created.
    
    Args:
        test_notes: List of clinical notes to process
        gold_data: Gold standard data for evaluation
        use_cache: Whether to use cached results if available
        overwrite_cache: Whether to force rerun of extractions
        data_sources: List of data sources to evaluate (direct, dspy, pipeline, validator, structured)
        
    Returns:
        Dict of results by approach
    """
    if data_sources is None:
        data_sources = ["direct", "dspy"]  # Default to original approaches
    
    results = {}
    for source in data_sources:
        if source in AVAILABLE_DATA_SOURCES:
            results[source] = []
    
    llm_cache_dir = LLM_CACHE_DIR
    os.makedirs(llm_cache_dir, exist_ok=True)
    
    # Define cache paths for each approach
    cache_paths = {
        "direct": os.path.join(llm_cache_dir, "llm_direct.jsonl"),
        "dspy": os.path.join(llm_cache_dir, "llm_dspy.jsonl"),
        "pipeline": os.path.join(llm_cache_dir, "llm_pipeline.jsonl"),
        "validator": os.path.join(llm_cache_dir, "llm_validator.jsonl"),
        "structured": os.path.join(llm_cache_dir, "llm_structured.jsonl")
    }
    
    # Check if any cache is missing or needs regeneration
    need_to_run = overwrite_cache
    if not need_to_run and use_cache:
        # Check if any requested source is missing cache
        for source in data_sources:
            if source in AVAILABLE_DATA_SOURCES and not os.path.exists(cache_paths[source]):
                print(f"⚠️ Cache for {source} not found, will run extractions.")
                need_to_run = True
                break
        
        if not need_to_run:
            print("✅ All requested cache files found. Using cached results.")
    
    # Print summary of approaches being evaluated
    approach_display_names = {
        "direct": "Direct LLM",
        "dspy": "DSPy",
        "pipeline": "Pipeline LLM",
        "validator": "Validator LLM",
        "structured": "Structured DSPy"
    }
    selected_approaches = [approach_display_names.get(source, source.capitalize()) 
                         for source in data_sources if source in AVAILABLE_DATA_SOURCES]
    print(f"\n📋 Evaluating {len(selected_approaches)} approach(es): {', '.join(selected_approaches)}")
    
    # Process each data source
    for source in data_sources:
        if source not in AVAILABLE_DATA_SOURCES:
            continue
        
        cache_path = cache_paths[source]
        use_existing_cache = use_cache and os.path.exists(cache_path) and not overwrite_cache

        if use_existing_cache:
            print(f"📂 Loading cached {source} extraction results...")
            source_results = load_from_jsonl(cache_path)
            print(f"✅ Loaded {len(source_results)} {source} predictions")
        else:
            # Generate predictions using Step 1 generators
            print(f"🔄 Generating new {source} predictions (this may take a while)...")
            
            # Import the generators from Step 1
            print("📥 Importing Step 1 generators...")
            import sys
            from importlib import import_module
            
            # Add Step 1 to path if needed
            step1_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Step_1_data_generation")
            if step1_path not in sys.path:
                sys.path.append(step1_path)
            
            # Import necessary components
            print(f"🔍 Setting up {source} generator...")
            import asyncio
            import openai
            from dotenv import load_dotenv
            
            # Load environment variables for API keys
            load_dotenv()
            openai.api_key = os.getenv('OPENAI_API_KEY')
            
            # Generate predictions with the appropriate method
            print(f"🚀 Running {source} extraction...")
            source_results = []
            
            try:
                if source == "direct":
                    from direct_llm_generator import process_batch
                    
                    # Process in batches
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    direct_batch_size = 10  # Default batch size for direct approach
                    for i in tqdm(range(0, len(test_notes), direct_batch_size), desc="Direct LLM processing"):
                        batch = test_notes[i:i+direct_batch_size]
                        batch_ner_results, _ = loop.run_until_complete(process_batch(batch, {
                            'model_name': 'gpt-4.1-nano',
                            'temperature': 0.1,
                            'max_tokens': 2000,
                            'batch_size': direct_batch_size
                        }))
                        source_results.extend(batch_ner_results)
                
                elif source == "dspy":
                    from dspy_generator import extract_entities_with_dspy
                    
                    try:
                        # Import and configure dspy if not already imported
                        import dspy
                        os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
                        
                        # Process one note at a time since DSPy handles batching internally
                        for note in tqdm(test_notes, desc="DSPy processing"):
                            ner_data, _ = extract_entities_with_dspy(note)
                            source_results.append(ner_data)
                    except ImportError as e:
                        print(f"⚠️ DSPy import error: {e}. Make sure DSPy is installed.")
                        raise
                
                elif source == "pipeline":
                    from pipeline_llm_generator import process_batch
                    
                    # Process in batches
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    pipeline_batch_size = 5  # Default batch size for pipeline approach (3 API calls per note)
                    for i in tqdm(range(0, len(test_notes), pipeline_batch_size), desc="Pipeline processing"):
                        batch = test_notes[i:i+pipeline_batch_size]
                        batch_ner_results, _ = loop.run_until_complete(process_batch(batch, {
                            'model_name': 'gpt-4.1-nano',
                            'temperature': 0.1,
                            'max_tokens': 2000,
                            'batch_size': pipeline_batch_size
                        }))
                        source_results.extend(batch_ner_results)
                
                elif source == "validator":
                    from validator_llm_generator import process_batch
                    
                    # Process in batches
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    validator_batch_size = 8  # Default batch size for validator approach (2 API calls per note)
                    for i in tqdm(range(0, len(test_notes), validator_batch_size), desc="Validator processing"):
                        batch = test_notes[i:i+validator_batch_size]
                        batch_ner_results, _ = loop.run_until_complete(process_batch(batch, {
                            'model_name': 'gpt-4.1-nano',
                            'temperature': 0.1,
                            'max_tokens': 2000,
                            'batch_size': validator_batch_size
                        }))
                        source_results.extend(batch_ner_results)
                
                elif source == "structured":
                    # Need to initialize DSPy for structured approach
                    import dspy
                    from structured_dspy_generator import process_batch
                    
                    # Configure DSPy
                    dspy.settings.configure(lm=dspy.LM(
                        model='gpt-4.1-nano',
                        temperature=0.1,
                        max_tokens=2000
                    ))
                    
                    # Process in batches
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    structured_batch_size = 5  # Default batch size for structured approach (complex processing)
                    for i in tqdm(range(0, len(test_notes), structured_batch_size), desc="Structured processing"):
                        batch = test_notes[i:i+structured_batch_size]
                        batch_ner_results, _ = loop.run_until_complete(process_batch(batch, {
                            'model_name': 'gpt-4.1-nano',
                            'temperature': 0.1,
                            'max_tokens': 2000,
                            'batch_size': structured_batch_size
                        }))
                        source_results.extend(batch_ner_results)
            
            except ImportError as e:
                print(f"❌ Error importing module for {source} generator: {str(e)}")
                print(f"⚠️ Make sure all required packages are installed for {source} approach.")
                source_results = []
            except Exception as e:
                print(f"❌ Error running {source} generator: {str(e)}")
                print(f"⚠️ Falling back to zero metrics for {source} approach.")
                source_results = []
            
            # Save to cache
            print(f"💾 Saving {source} predictions to cache...")
            save_to_jsonl(source_results, cache_path)
            
            print(f"✅ Generated {len(source_results)} {source} predictions")
        
        # Compute metrics for each note
        print(f"📊 Computing metrics for {source} approach...")
        for idx, gold in enumerate(tqdm(gold_data, desc=f"Evaluating {source} approach")):
            # Check if we have a prediction for this note
            if idx < len(source_results):
                pred_entities = source_results[idx].get("entities", [])
            else:
                pred_entities = []
            gold_entities = gold.get("entities", [])
            results[source].append({"overall": entity_level_metrics(pred_entities, gold_entities)})

    # Return the results for all evaluated approaches
    return results

def get_display_name(source):
    """Get a user-friendly display name for a source."""
    approach_display_names = {
        "direct": "Direct LLM",
        "dspy": "DSPy",
        "pipeline": "Pipeline LLM",
        "validator": "Validator LLM",
        "structured": "Structured DSPy"
    }
    return approach_display_names.get(source, source.capitalize())

# Main function for stand-alone execution
if __name__ == "__main__":
    print("LLM Approach Evaluator - Run through run_step3.py") 