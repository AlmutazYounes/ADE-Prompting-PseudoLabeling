#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3: Model Evaluation
-----------------------
This script evaluates the trained BERT models against the gold standard data
and compares their performance with direct LLM and DSPy approaches.

The script produces performance metrics and visualizations that are saved to
the analysis folder.
"""

import os
import json
import shutil
import logging
from datetime import datetime
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

from transformers import AutoModelForTokenClassification, AutoTokenizer
import evaluate

# Import local config
from Step_3_model_evaluation.config import (
    STEP_2_GOLD_NER_DATA, 
    MAX_TEST_NOTES, 
    TRAINED_MODELS_DIR, 
    LLM_CACHE_DIR,
    COMPARISON_RESULTS_DIR, 
    BERT_MAX_LENGTH, 
    DEFAULT_BERT_MODEL_NAME,
    NER_LABELS,
    ID_TO_LABEL,
    VISUALIZATION,
    USE_CACHE_DEFAULT,
    OVERWRITE_CACHE_DEFAULT
)

# Suppress unnecessary warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore")
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

#############################
# Utility Functions
#############################

def get_device():
    """
    Determine the best available device for model inference.
    
    Returns:
        torch.device: The device to use
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

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

def format_time(seconds):
    """Format time in seconds to a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def get_entity_distribution(data):
    """Get the distribution of entity types in a dataset."""
    distribution = {}
    
    for item in data:
        entities = item.get("entities", [])
        for entity in entities:
            label = entity.get("label")
            if label:
                distribution[label] = distribution.get(label, 0) + 1
    
    return distribution

#############################
# Model Evaluation Functions
#############################

def find_all_models(bert_output_dir=TRAINED_MODELS_DIR):
    """
    Find all trained model subfolders in the given directory.
    Returns a dict: {model_label: (model, tokenizer, path)}
    """
    models = {}
    if not os.path.exists(bert_output_dir):
        print(f"⚠️ Model directory does not exist: {bert_output_dir}")
        return models
    
    print(f"\n{'='*70}")
    print(f"🔍 Scanning for trained models in: {bert_output_dir}")
    
    for subdir in os.listdir(bert_output_dir):
        full_path = os.path.join(bert_output_dir, subdir)
        if os.path.isdir(full_path):
            try:
                model = AutoModelForTokenClassification.from_pretrained(full_path)
                tokenizer = AutoTokenizer.from_pretrained(full_path)
                # Use folder name as label, e.g. direct_Bio_ClinicalBERT
                models[subdir] = (model, tokenizer, full_path)
                print(f"✅ Loaded model: {subdir}")
            except Exception as e:
                print(f"❌ Error loading model from {full_path}: {e}")
    
    print(f"🔢 Found {len(models)} trained models")
    print(f"{'='*70}\n")
    return models

def evaluate_bert_model(model, tokenizer, gold_data):
    """
    Evaluate a BERT model using token-level BIO evaluation.
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer for the model
        gold_data: Gold standard data for evaluation
        
    Returns:
        List of metric dictionaries
    """
    seqeval = evaluate.load("seqeval")
    device = get_device()
    print(f"\033[1;32m[INFO] Using device: {device}\033[0m")
    
    model.to(device)
    model.eval()
    
    all_pred_tags = []
    all_gold_tags = []
    
    for idx, gold in enumerate(tqdm(gold_data, desc="Evaluating BERT model", ncols=100)):
        text = gold["text"]
        
        # Tokenize for model input
        model_inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=BERT_MAX_LENGTH, 
            padding="max_length"
        )
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        
        # Tokenize for alignment
        alignment_encoding = tokenizer(
            text, 
            return_offsets_mapping=True, 
            add_special_tokens=False,
            truncation=True,
            max_length=BERT_MAX_LENGTH - 2  # Account for [CLS] and [SEP] tokens
        )
        tokens = tokenizer.convert_ids_to_tokens(alignment_encoding.input_ids)
        offset_mapping = alignment_encoding.offset_mapping
        
        # Get model predictions
        with torch.no_grad():
            outputs = model(**model_inputs)
            predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()
        
        # Extract predictions for actual tokens (skip special tokens)
        actual_predictions = predictions[1:len(tokens)+1]
        
        # Convert predictions to labels
        pred_tags = []
        for pred in actual_predictions:
            label = model.config.id2label.get(int(pred), 'O')
            pred_tags.append(label)
        
        # Generate gold tags using the same tokenization
        gold_entities = gold.get('entities', [])
        gold_tags = ['O'] * len(tokens)
        
        # Assign BIO tags based on entity spans
        for entity in gold_entities:
            entity_start = entity.get('start')
            entity_end = entity.get('end')
            entity_label = entity.get('label')
            
            for i, (start, end) in enumerate(offset_mapping):
                if end <= entity_start or start >= entity_end:
                    continue
                
                if i == 0 or offset_mapping[i-1][1] <= entity_start:
                    gold_tags[i] = f'B-{entity_label}'
                else:
                    gold_tags[i] = f'I-{entity_label}'
        
        # Ensure sequences have same length
        min_len = min(len(pred_tags), len(gold_tags))
        pred_tags = pred_tags[:min_len]
        gold_tags = gold_tags[:min_len]
        
        all_pred_tags.append(pred_tags)
        all_gold_tags.append(gold_tags)
    
    # Compute metrics
    results = seqeval.compute(predictions=all_pred_tags, references=all_gold_tags)
    
    # Print nicely formatted results
    print("\n" + "=" * 50)
    print(f"📊 BERT MODEL EVALUATION RESULTS")
    print("-" * 50)
    print(f"F1 Score:    {results.get('overall_f1', 0):.4f}")
    print(f"Precision:   {results.get('overall_precision', 0):.4f}")
    print(f"Recall:      {results.get('overall_recall', 0):.4f}")
    print("=" * 50 + "\n")
    
    # Return list of metric dictionaries for compatibility
    return [{
        'overall': {
            'precision': results.get('overall_precision', 0),
            'recall': results.get('overall_recall', 0),
            'f1': results.get('overall_f1', 0),
        }
    } for _ in range(len(all_pred_tags))]

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

def evaluate_llm_approaches(test_notes, gold_data, use_cache=USE_CACHE_DEFAULT, overwrite_cache=OVERWRITE_CACHE_DEFAULT):
    """
    Evaluate LLM-based approaches (Direct and DSPy).
    Uses the actual implementations from Step 1 when cache needs to be created.
    """
    results = {"direct": [], "dspy": []}
    llm_cache_dir = LLM_CACHE_DIR
    os.makedirs(llm_cache_dir, exist_ok=True)
    llm_direct_path = os.path.join(llm_cache_dir, "llm_direct.jsonl")
    llm_dspy_path = os.path.join(llm_cache_dir, "llm_dspy.jsonl")

    # Check if we need to generate predictions or can use cache
    use_existing_cache = use_cache and os.path.exists(llm_direct_path) and os.path.exists(llm_dspy_path) and not overwrite_cache

    if use_existing_cache:
        print("📂 Loading cached LLM extraction results...")
        direct_ner = load_from_jsonl(llm_direct_path)
        dspy_ner = load_from_jsonl(llm_dspy_path)
        print(f"✅ Loaded {len(direct_ner)} Direct LLM predictions")
        print(f"✅ Loaded {len(dspy_ner)} DSPy predictions")
    else:
        # We need to generate predictions using Step 1 generators
        print("🔄 Generating new LLM predictions (this may take a while)...")
        
        # Import the generators from Step 1
        print("📥 Importing Step 1 generators...")
        import sys
        from importlib import import_module
        
        # Add Step 1 to path if needed
        step1_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Step_1_data_generation")
        if step1_path not in sys.path:
            sys.path.append(step1_path)
        
        try:
            # Import components from direct_llm_generator
            print("🔍 Setting up Direct LLM generator...")
            import asyncio
            import openai
            from dotenv import load_dotenv
            
            # Load environment variables for API keys
            load_dotenv()
            openai.api_key = os.getenv('OPENAI_API_KEY')
            
            # Import the direct LLM functions
            from direct_llm_generator import create_entities, process_batch, PROMPT_TEMPLATE, LLM_MODEL_NAME, LLM_TEMPERATURE, LLM_MAX_TOKENS, BATCH_SIZE
            
            # Import DSPy generator
            print("🔍 Setting up DSPy generator...")
            from dspy_generator import extract_entities_with_dspy, EnsembleADEExtractor
            
            # Generate predictions with Direct LLM
            print("🚀 Running Direct LLM extraction...")
            direct_ner = []
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Process in batches
            for i in tqdm(range(0, len(test_notes), BATCH_SIZE), desc="Direct LLM processing"):
                batch = test_notes[i:i+BATCH_SIZE]
                batch_ner_results, _ = loop.run_until_complete(process_batch(batch))
                direct_ner.extend(batch_ner_results)
            
            # Generate predictions with DSPy
            print("🚀 Running DSPy extraction...")
            dspy_ner = []
            for note in tqdm(test_notes, desc="DSPy processing"):
                ner_data, _ = extract_entities_with_dspy(note)
                dspy_ner.append(ner_data)
            
            # Save to cache
            print("💾 Saving predictions to cache...")
            save_to_jsonl(direct_ner, llm_direct_path)
            save_to_jsonl(dspy_ner, llm_dspy_path)
            
            print(f"✅ Generated {len(direct_ner)} Direct LLM predictions")
            print(f"✅ Generated {len(dspy_ner)} DSPy predictions")
            
        except Exception as e:
            print(f"❌ Error running LLM generators: {str(e)}")
            print("⚠️ Falling back to zero metrics")
            # Return zero metrics
            for _ in range(len(gold_data)):
                results["direct"].append({"overall": {"precision": 0, "recall": 0, "f1": 0}})
                results["dspy"].append({"overall": {"precision": 0, "recall": 0, "f1": 0}})
            return results

    # Compute metrics for each note
    print("📊 Computing metrics for LLM approaches...")
    for idx, gold in enumerate(tqdm(gold_data, desc="Evaluating LLM approaches")):
        # Direct LLM
        if idx < len(direct_ner):
            pred_entities = direct_ner[idx].get("entities", [])
        else:
            pred_entities = []
        gold_entities = gold.get("entities", [])
        results["direct"].append({"overall": entity_level_metrics(pred_entities, gold_entities)})

        # DSPy
        if idx < len(dspy_ner):
            pred_entities = dspy_ner[idx].get("entities", [])
        else:
            pred_entities = []
        results["dspy"].append({"overall": entity_level_metrics(pred_entities, gold_entities)})
    
    return results

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

#############################
# Visualization Functions
#############################

def save_metrics_json(metrics, results_dir):
    """Save metrics as a JSON file."""
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    return metrics_path

def create_f1_comparison(metrics, results_dir):
    """Create and save a bar chart comparing F1 scores across approaches."""
    # Convert metrics to DataFrame
    df = pd.DataFrame([
        {"Approach": approach, "F1 Score": metrics_dict["f1"]} 
        for approach, metrics_dict in metrics.items()
    ])
    
    # Create plot
    plt.figure(figsize=VISUALIZATION["figsize_default"])
    sns.barplot(x="Approach", y="F1 Score", data=df)
    plt.title("F1 Score Comparison Across Approaches")
    plt.ylim(0, 1.0)
    plt.ylabel("F1 Score")
    plt.xticks(rotation=15)
    plt.tight_layout()
    
    # Save figure
    f1_path = os.path.join(results_dir, "f1_comparison.png")
    plt.savefig(f1_path, dpi=VISUALIZATION["dpi"])
    plt.close()
    
    return f1_path

def create_prf_comparison(metrics, results_dir):
    """Create and save a grouped bar chart comparing precision, recall, and F1."""
    # Melt the data for grouped bars
    df_list = []
    for approach, metrics_dict in metrics.items():
        for metric, value in metrics_dict.items():
            df_list.append({
                "Approach": approach,
                "Metric": metric.capitalize(),
                "Score": value
            })
    
    df = pd.DataFrame(df_list)
    
    # Create plot
    plt.figure(figsize=VISUALIZATION["figsize_default"])
    sns.barplot(x="Approach", y="Score", hue="Metric", data=df)
    plt.title("Precision, Recall, F1 Comparison Across Approaches")
    plt.ylim(0, 1.0)
    plt.xticks(rotation=15)
    plt.legend(title="Metric")
    plt.tight_layout()
    
    # Save figure
    prf_path = os.path.join(results_dir, "prf_comparison.png")
    plt.savefig(prf_path, dpi=VISUALIZATION["dpi"])
    plt.close()
    
    return prf_path

def create_summary_report(metrics, results_dir):
    """Create and save a text report summarizing the evaluation results."""
    report_path = os.path.join(results_dir, "summary_report.txt")
    
    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("BIO_CLINICALBERT MODEL EVALUATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall performance table
        f.write("PERFORMANCE COMPARISON:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Approach':<25} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}\n")
        f.write("-" * 80 + "\n")
        
        for approach, metrics_dict in sorted(metrics.items()):
            precision = metrics_dict.get("precision", 0)
            recall = metrics_dict.get("recall", 0)
            f1 = metrics_dict.get("f1", 0)
            
            f.write(f"{approach:<25} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}\n")
        
        f.write("\n\n")
        
        # Best model
        best_approach = max(metrics.items(), key=lambda x: x[1]["f1"])[0]
        best_f1 = metrics[best_approach]["f1"]
        
        f.write("BEST PERFORMING MODEL:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Approach: {best_approach}\n")
        f.write(f"F1 Score: {best_f1:.4f}\n")
        f.write(f"Precision: {metrics[best_approach]['precision']:.4f}\n")
        f.write(f"Recall: {metrics[best_approach]['recall']:.4f}\n")
        
        f.write("\n\n")
        f.write("=" * 80 + "\n")
    
    return report_path

def analyze_error_patterns(predictions, gold_data, output_dir):
    """Analyze error patterns in model predictions."""
    # Initialize counters
    false_positives = Counter()
    false_negatives = Counter()
    true_positives = Counter()
    
    # Analyze each example
    for pred, gold in zip(predictions, gold_data):
        pred_entities = {(e["start"], e["end"], e["label"]) for e in pred.get("entities", [])}
        gold_entities = {(e["start"], e["end"], e["label"]) for e in gold.get("entities", [])}
        
        # Find false positives (in pred but not in gold)
        for entity in pred_entities:
            if entity not in gold_entities:
                label = entity[2]
                false_positives[label] += 1
        
        # Find false negatives (in gold but not in pred)
        for entity in gold_entities:
            if entity not in pred_entities:
                label = entity[2]
                false_negatives[label] += 1
            else:
                # True positives
                label = entity[2]
                true_positives[label] += 1
    
    # Create results dictionary
    results = {
        "false_positives": dict(false_positives),
        "false_negatives": dict(false_negatives),
        "true_positives": dict(true_positives)
    }
    
    # Calculate entity-level metrics
    entity_metrics = {}
    all_labels = set(list(true_positives.keys()) + 
                     list(false_positives.keys()) + 
                     list(false_negatives.keys()))
    
    for label in all_labels:
        tp = true_positives.get(label, 0)
        fp = false_positives.get(label, 0)
        fn = false_negatives.get(label, 0)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        entity_metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn
        }
    
    results["entity_metrics"] = entity_metrics
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "error_analysis.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Create visualization
    plot_error_analysis(results, output_dir)
    
    return results

def plot_error_analysis(results, output_dir):
    """Create charts visualizing error analysis results."""
    # Entity-level metrics chart
    if "entity_metrics" in results:
        metrics = results["entity_metrics"]
        
        # Convert to DataFrame
        df_list = []
        for label, values in metrics.items():
            for metric_name, value in values.items():
                if metric_name != "support":  # Exclude support from the chart
                    df_list.append({
                        "Entity": label,
                        "Metric": metric_name.capitalize(),
                        "Value": value
                    })
        
        df = pd.DataFrame(df_list)
        
        # Create plot
        plt.figure(figsize=VISUALIZATION["figsize_default"])
        sns.barplot(x="Entity", y="Value", hue="Metric", data=df)
        plt.title("Performance Metrics by Entity Type")
        plt.ylim(0, 1.0)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, "entity_metrics.png"), dpi=VISUALIZATION["dpi"])
        plt.close()
    
    # Error distribution chart
    fp = results.get("false_positives", {})
    fn = results.get("false_negatives", {})
    
    # Convert to DataFrame
    df_list = []
    for label in set(list(fp.keys()) + list(fn.keys())):
        df_list.append({
            "Entity": label,
            "Error Type": "False Positives",
            "Count": fp.get(label, 0)
        })
        df_list.append({
            "Entity": label,
            "Error Type": "False Negatives",
            "Count": fn.get(label, 0)
        })
    
    df = pd.DataFrame(df_list)
    
    # Create plot
    plt.figure(figsize=VISUALIZATION["figsize_default"])
    sns.barplot(x="Entity", y="Count", hue="Error Type", data=df)
    plt.title("Error Distribution by Entity Type")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, "error_distribution.png"), dpi=VISUALIZATION["dpi"])
    plt.close()

#############################
# Main Function
#############################

def run_evaluation(config):
    """
    Run the complete model evaluation pipeline using a config dict.
    Keys expected in config:
        - gold_data_path
        - max_notes
        - output_dir
        - name
        - use_cache
        - overwrite_cache
        - skip_llm
        - detailed_analysis
        - data_sources (list of sources to evaluate)
    """
    # Print banner
    print_banner("STEP 3: MODEL EVALUATION")
    
    # Generate output directory name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = config.get("name")
    if name:
        dir_name = f"{timestamp}_{name}"
    else:
        dir_name = f"{timestamp}_bio_clinicalbert"
    
    results_dir = config.get("output_dir") if config.get("output_dir") else os.path.join(COMPARISON_RESULTS_DIR, dir_name)
    os.makedirs(results_dir, exist_ok=True)
    print(f"📊 Results will be saved to: {results_dir}")
    
    # Load gold standard data
    gold_data_path = config.get("gold_data_path", STEP_2_GOLD_NER_DATA)
    max_notes = config.get("max_notes", MAX_TEST_NOTES)
    print(f"\n{'='*70}")
    print(f"📂 Loading gold standard data from {gold_data_path}")
    gold_data = load_from_jsonl(gold_data_path)
    
    if max_notes is not None:
        gold_data = gold_data[:max_notes]
    
    test_notes = [entry["text"] for entry in gold_data if "text" in entry]
    print(f"✅ Using {len(test_notes)} gold standard notes for evaluation")
    print(f"{'='*70}\n")
    
    # Get data sources to evaluate
    data_sources = config.get("data_sources", [])
    if not data_sources:
        print("⚠️ No data sources specified for evaluation. Exiting.")
        return 1
    print(f"🔎 Evaluating data sources: {data_sources}")
    
    # Load BERT models
    print_banner("LOADING BERT MODELS")
    models = find_all_models()
    
    # Initialize results dictionary
    all_results = {}
    
    # Evaluate all found models that match the selected data sources
    print_banner("EVALUATING BERT MODELS")
    for model_label, (model, tokenizer, _) in models.items():
        # Only evaluate models whose label starts with one of the selected data sources
        if any(model_label.startswith(source) for source in data_sources):
            print(f"\n{'='*70}")
            print(f"🔍 Evaluating {model_label}")
            print(f"{'='*70}")
            results = evaluate_bert_model(model, tokenizer, gold_data)
            all_results[model_label] = results
        else:
            print(f"⏩ Skipping model {model_label} (not in selected data sources)")
    
    # Evaluate LLM approaches if not skipped and if selected
    if not config.get("skip_llm", False):
        for source in data_sources:
            if source in ["direct", "dspy"]:
                print_banner(f"EVALUATING LLM APPROACH: {source}")
                llm_results = evaluate_llm_approaches(
                    test_notes, 
                    gold_data,
                    use_cache=config.get("use_cache", USE_CACHE_DEFAULT),
                    overwrite_cache=config.get("overwrite_cache", OVERWRITE_CACHE_DEFAULT)
                )
                # Only add the selected LLM approach
                if source == "direct":
                    all_results["Direct LLM"] = llm_results["direct"]
                elif source == "dspy":
                    all_results["DSPy"] = llm_results["dspy"]
    
    # Calculate overall metrics
    print_banner("CALCULATING OVERALL METRICS")
    metrics = {k: calculate_overall_metrics(v) for k, v in all_results.items()}
    
    # Save metrics as JSON
    print(f"💾 Saving metrics to {results_dir}")
    save_metrics_json(metrics, results_dir)
    
    # Create visualizations
    print_banner("GENERATING VISUALIZATIONS")
    create_f1_comparison(metrics, results_dir)
    create_prf_comparison(metrics, results_dir)
    create_summary_report(metrics, results_dir)
        
    # Detailed error analysis if requested
    if config.get("detailed_analysis", False):
        print_banner("PERFORMING ERROR ANALYSIS")
        # Implement error analysis here if needed
        print("⚠️ Detailed error analysis requested but not implemented in this version")
    
    # Print summary
    print_banner("EVALUATION SUMMARY")
    print("\n" + "=" * 80)
    print(f"{'Approach':<25} {'F1':<10} {'Precision':<10} {'Recall':<10}")
    print("-" * 80)
    for approach, metrics_dict in sorted(metrics.items(), key=lambda x: x[1]['f1'], reverse=True):
        print(f"{approach:<25} {metrics_dict['f1']:<10.4f} {metrics_dict['precision']:<10.4f} {metrics_dict['recall']:<10.4f}")
    print("=" * 80 + "\n")
    
    print(f"✅ Evaluation complete. Results saved to: {results_dir}")
    
    return 0

if __name__ == "__main__":
    # Example config dict (edit as needed)
    config = {
        "gold_data_path": STEP_2_GOLD_NER_DATA,
        "max_notes": MAX_TEST_NOTES,
        "output_dir": None,  # or specify a path
        "name": None,        # or specify a name
        "use_cache": True,
        "overwrite_cache": False,
        "skip_llm": False,
        "detailed_analysis": False,
        "data_sources": ["direct", "dspy"]
    }
    exit(run_evaluation(config)) 