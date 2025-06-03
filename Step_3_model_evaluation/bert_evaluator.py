#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3: BERT Model Evaluation
---------------------------
This script evaluates the trained BERT models against the gold standard data.

The script produces performance metrics for BERT models that are saved to
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

from transformers import AutoModelForTokenClassification, AutoTokenizer
import evaluate

# Suppress unnecessary warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore")
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

# Constants
STEP_2_GOLD_NER_DATA = os.path.join("Step_1_data_generation", "data", "gold", "gold_ner_data.jsonl")
TRAINED_MODELS_DIR = os.path.join("Step_2_train_BERT_models", "trained_models")
BERT_MAX_LENGTH = 256
DEFAULT_BERT_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
NER_LABELS = ["ADE", "Drug", "Dosage", "Route", "Frequency", "Duration", "Reason", "Form"]
ID_TO_LABEL = {
    0: "O", 
    1: "B-ADE", 2: "I-ADE",
    3: "B-Drug", 4: "I-Drug",
    5: "B-Dosage", 6: "I-Dosage",
    7: "B-Route", 8: "I-Route",
    9: "B-Frequency", 10: "I-Frequency",
    11: "B-Duration", 12: "I-Duration",
    13: "B-Reason", 14: "I-Reason",
    15: "B-Form", 16: "I-Form"
}

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

def print_banner(text):
    """Print a nicely formatted banner with the given text."""
    term_width = shutil.get_terminal_size((80, 20)).columns
    banner_text = f" {text} "
    banner = f"\033[1;44m{banner_text.center(term_width)}\033[0m"
    print("\n" + banner + "\n")

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

def evaluate_bert_models(gold_data, data_sources):
    """
    Evaluate all BERT models that match the selected data sources.
    
    Args:
        gold_data: Gold standard data for evaluation
        data_sources: List of data source prefixes to match
        
    Returns:
        Dictionary mapping model names to evaluation results
    """
    # Load BERT models
    print_banner("LOADING BERT MODELS")
    models = find_all_models()
    
    # Initialize results dictionary
    results = {}
    
    # Evaluate all found models that match the selected data sources
    print_banner("EVALUATING BERT MODELS")
    for model_label, (model, tokenizer, _) in models.items():
        # Only evaluate models whose label starts with one of the selected data sources
        if any(model_label.startswith(source) for source in data_sources):
            print(f"\n{'='*70}")
            print(f"🔍 Evaluating {model_label}")
            print(f"{'='*70}")
            model_results = evaluate_bert_model(model, tokenizer, gold_data)
            results[model_label] = model_results
        else:
            print(f"⏩ Skipping model {model_label} (not in selected data sources)")
    
    return results

# Main function for stand-alone execution
if __name__ == "__main__":
    print("BERT Model Evaluator - Run through run_step3.py") 