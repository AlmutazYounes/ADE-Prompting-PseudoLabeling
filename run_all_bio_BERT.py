#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive evaluation script to compare Bio_ClinicalBERT ADE extraction approaches:
1. Bio_ClinicalBERT trained on direct LLM output
2. Bio_ClinicalBERT trained on DSPy output
3. Direct LLM approach
4. DSPy approach

This script loads the gold standard dataset and evaluates each approach,
producing comparative metrics and visualizations.
"""

import os
import json
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from transformers import AutoModelForTokenClassification, AutoTokenizer
from tqdm import tqdm
from datetime import datetime
import logging

from utils.config import (
    GOLD_STANDARD_PATH, MAX_TEST_NOTES,
    BERT_OUTPUT_DIR, BERT_MODEL_NAME, BERT_MAX_LENGTH,
    STEP_2_GOLD_NER_DATA
)
from utils.data_transformation import load_from_jsonl
from utils.extraction import initialize_extractor
from utils.evaluation import calculate_entity_metrics
from utils.dataset import ADEDatasetProcessor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_latest_bio_clinicalbert_models():
    """Find and load the latest Bio_ClinicalBERT models for both approaches."""
    direct_models = glob.glob(os.path.join(BERT_OUTPUT_DIR, "direct_approach_*"))
    dspy_models = glob.glob(os.path.join(BERT_OUTPUT_DIR, "dspy_approach_*"))
    models = {"direct": None, "dspy": None}
    
    if direct_models:
        latest_direct = sorted(direct_models, key=os.path.getctime, reverse=True)[0]
        model_path = os.path.join(latest_direct, "model")
        
        # Check if model directory exists, if not use the parent directory
        if not os.path.exists(model_path):
            model_path = latest_direct
            
        try:
            # Try to load the model config first to check if it exists
            config_path = os.path.join(latest_direct, "model_config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    model_config = json.load(f)
                # Initialize base model with label mappings
                model = AutoModelForTokenClassification.from_pretrained(
                    BERT_MODEL_NAME, 
                    num_labels=len(model_config["id2label"]), 
                    id2label=model_config["id2label"],
                    label2id=model_config["label2id"]
                )
            else:
                # Fall back to loading a saved model if it exists
                model = AutoModelForTokenClassification.from_pretrained(model_path)
                
            tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
            models["direct"] = (model, tokenizer, latest_direct)
            logger.info(f"Loaded Bio_ClinicalBERT (direct) from: {latest_direct}")
        except Exception as e:
            logger.error(f"Error loading Bio_ClinicalBERT (direct): {e}")
    else:
        logger.warning("No Bio_ClinicalBERT direct models found")
        
    if dspy_models:
        latest_dspy = sorted(dspy_models, key=os.path.getctime, reverse=True)[0]
        model_path = os.path.join(latest_dspy, "model")
        
        # Check if model directory exists, if not use the parent directory
        if not os.path.exists(model_path):
            model_path = latest_dspy
            
        try:
            # Try to load the model config first to check if it exists
            config_path = os.path.join(latest_dspy, "model_config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    model_config = json.load(f)
                # Initialize base model with label mappings
                model = AutoModelForTokenClassification.from_pretrained(
                    BERT_MODEL_NAME, 
                    num_labels=len(model_config["id2label"]), 
                    id2label=model_config["id2label"],
                    label2id=model_config["label2id"]
                )
            else:
                # Fall back to loading a saved model if it exists
                model = AutoModelForTokenClassification.from_pretrained(model_path)
                
            tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
            models["dspy"] = (model, tokenizer, latest_dspy)
            logger.info(f"Loaded Bio_ClinicalBERT (dspy) from: {latest_dspy}")
        except Exception as e:
            logger.error(f"Error loading Bio_ClinicalBERT (dspy): {e}")
    else:
        logger.warning("No Bio_ClinicalBERT dspy models found")
        
    return models

def evaluate_bio_clinicalbert_model(model, tokenizer, test_texts, gold_data):
    """Evaluate a Bio_ClinicalBERT model on the gold standard."""
    from torch.utils.data import DataLoader
    from utils.dataset import ADEDataset
    
    # Determine device based on availability
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                         "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    model.to(device)
    model.eval()
    
    # Process raw data to get predictions
    results = []
    
    for idx, gold in enumerate(tqdm(gold_data, desc="Evaluating Bio_ClinicalBERT")):
        text = gold["text"]
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", truncation=True, 
                          max_length=BERT_MAX_LENGTH, padding="max_length")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()
            
        # Debug prediction distribution
        if idx < 5:
            pred_labels = [model.config.id2label.get(p.item()) for p in predictions if p.item() in model.config.id2label]
            label_counts = {}
            for label in pred_labels:
                if label:
                    label_counts[label] = label_counts.get(label, 0) + 1
            logger.info(f"Prediction distribution: {label_counts}")
            
        # Convert predictions to entities
        entities = []
        current_entity = None
        
        # Get token offsets
        encoding = tokenizer(text, return_offsets_mapping=True)
        offset_mapping = encoding.offset_mapping
        
        for i, (pred, (start, end)) in enumerate(zip(predictions, offset_mapping)):
            # Skip special tokens
            if start == end:
                continue
                
            # Get the predicted label - safely handle missing label IDs
            label = model.config.id2label.get(pred.item())
            if label is None:
                continue
                
            # Handle entity boundaries
            if label.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                    
                entity_type = label[2:]  # Remove "B-" prefix
                current_entity = {
                    "start": start,
                    "end": end,
                    "label": entity_type
                }
            elif label.startswith("I-") and current_entity and current_entity["label"] == label[2:]:
                # Extend the current entity
                current_entity["end"] = end
            elif label == "O" and current_entity:
                entities.append(current_entity)
                current_entity = None
                
        # Add the last entity if there is one
        if current_entity:
            entities.append(current_entity)
            
        # Create a record with predictions
        pred_record = {
            "text": text,
            "entities": entities
        }
        
        # Convert entities to drugs/adverse_events format for metric calculation
        drugs = []
        adverse_events = []
        
        for entity in entities:
            entity_text = text[entity["start"]:entity["end"]]
            if entity["label"] == "DRUG":
                drugs.append(entity_text)
            elif entity["label"] == "ADE":
                adverse_events.append(entity_text)
        
        pred_record_converted = {
            "text": text,
            "drugs": drugs,
            "adverse_events": adverse_events
        }
        
        # Also convert gold data if it's in entity format
        gold_converted = gold
        if "entities" in gold and "drugs" not in gold:
            gold_drugs = []
            gold_adverse_events = []
            
            for entity in gold["entities"]:
                entity_text = text[entity["start"]:entity["end"]]
                if entity["label"] == "DRUG":
                    gold_drugs.append(entity_text)
                elif entity["label"] == "ADE":
                    gold_adverse_events.append(entity_text)
            
            gold_converted = {
                "text": text,
                "drugs": gold_drugs,
                "adverse_events": gold_adverse_events
            }
        
        # Calculate metrics
        metrics = calculate_entity_metrics(pred_record_converted, gold_converted)
        
        # Debug information
        if idx < 5:  # Only print for first few examples
            logger.info(f"\nExample {idx+1}:")
            logger.info(f"Text: {text[:100]}...")
            logger.info(f"Predicted drugs: {drugs}")
            logger.info(f"Predicted ADEs: {adverse_events}")
            if "drugs" in gold_converted:
                logger.info(f"Gold drugs: {gold_converted['drugs']}")
                logger.info(f"Gold ADEs: {gold_converted['adverse_events']}")
            else:
                logger.info(f"Gold entities: {gold['entities']}")
            logger.info(f"Metrics: P={metrics['overall']['precision']:.4f}, R={metrics['overall']['recall']:.4f}, F1={metrics['overall']['f1']:.4f}")
            logger.info(f"True positives: {metrics['overall']['true_positives']}, False positives: {metrics['overall']['false_positives']}, False negatives: {metrics['overall']['false_negatives']}")
            
        results.append(metrics)
        
    # Calculate overall metrics
    overall_precision = sum(r['overall']['precision'] for r in results) / len(results) if results else 0
    overall_recall = sum(r['overall']['recall'] for r in results) / len(results) if results else 0
    overall_f1 = sum(r['overall']['f1'] for r in results) / len(results) if results else 0
    total_tp = sum(r['overall']['true_positives'] for r in results)
    total_fp = sum(r['overall']['false_positives'] for r in results)
    total_fn = sum(r['overall']['false_negatives'] for r in results)
    
    logger.info("\n========== BERT MODEL EVALUATION SUMMARY ==========")
    logger.info(f"Evaluated {len(results)} examples")
    logger.info(f"Overall Precision: {overall_precision:.4f}")
    logger.info(f"Overall Recall: {overall_recall:.4f}")
    logger.info(f"Overall F1: {overall_f1:.4f}")
    logger.info(f"Total true positives: {total_tp}")
    logger.info(f"Total false positives: {total_fp}")
    logger.info(f"Total false negatives: {total_fn}")
    logger.info("================================================\n")
        
    return results

def evaluate_llm_and_dspy(test_notes, gold_data):
    """Evaluate both LLM-based approaches: Direct and DSPy using pipeline logic."""
    results = {"direct": [], "dspy": []}
    
    # Direct approach
    logger.info("Extracting ADEs using Direct approach (pipeline logic)...")
    direct_extractor = initialize_extractor(mode="direct")
    direct_processor = ADEDatasetProcessor(extractor=direct_extractor)
    direct_extracted = direct_processor.extract_ades_batched(test_notes)
    
    # Process into NER format for evaluation
    direct_ner = direct_processor.prepare_ner_data(direct_extracted)
    
    for record, gold in zip(direct_ner, gold_data):
        metrics = calculate_entity_metrics(record, gold)
        results["direct"].append(metrics)
    
    # DSPy approach
    logger.info("Extracting ADEs using DSPy approach (pipeline logic)...")
    dspy_extractor = initialize_extractor(mode="dspy")
    dspy_processor = ADEDatasetProcessor(extractor=dspy_extractor)
    dspy_extracted = dspy_processor.extract_ades_batched(test_notes)
    
    # Process into NER format for evaluation
    dspy_ner = dspy_processor.prepare_ner_data(dspy_extracted)
    
    for record, gold in zip(dspy_ner, gold_data):
        metrics = calculate_entity_metrics(record, gold)
        results["dspy"].append(metrics)
        
    return results

def calculate_overall_metrics(metrics_list):
    if not metrics_list:
        return {"precision": 0, "recall": 0, "f1": 0}
        
    precision = sum(m["overall"]["precision"] for m in metrics_list) / len(metrics_list)
    recall = sum(m["overall"]["recall"] for m in metrics_list) / len(metrics_list)
    f1 = sum(m["overall"]["f1"] for m in metrics_list) / len(metrics_list)
    
    return {"precision": precision, "recall": recall, "f1": f1}

def visualize_results(all_results, results_dir):
    # all_results: dict with keys 'Bio_ClinicalBERT (Direct)', 'Bio_ClinicalBERT (DSPy)', 'Direct LLM', 'DSPy'
    os.makedirs(results_dir, exist_ok=True)
    metrics = {k: calculate_overall_metrics(v) for k, v in all_results.items()}
    
    # Save metrics as JSON
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Plot
    df = pd.DataFrame([
        {"Approach": k, **v} for k, v in metrics.items()
    ])
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Approach", y="f1", data=df)
    plt.title("F1 Score Comparison Across Approaches")
    plt.ylim(0, 1.0)
    plt.ylabel("F1 Score")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "f1_comparison.png"))
    
    # Also plot precision and recall
    plt.figure(figsize=(10, 6))
    df_melt = df.melt(id_vars=["Approach"], value_vars=["precision", "recall", "f1"], 
                     var_name="Metric", value_name="Score")
    sns.barplot(x="Approach", y="Score", hue="Metric", data=df_melt)
    plt.title("Precision, Recall, F1 Comparison Across Approaches")
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "prf_comparison.png"))
    
    logger.info(f"Results saved to: {results_dir}")

def main():
    # Load gold standard data - use NER format
    logger.info(f"Loading gold standard NER data from {STEP_2_GOLD_NER_DATA}")
    gold_data = load_from_jsonl(STEP_2_GOLD_NER_DATA)
    gold_data = gold_data[:MAX_TEST_NOTES]
    test_notes = [entry["text"] for entry in gold_data if "text" in entry]
    
    logger.info(f"Using {len(test_notes)} gold standard notes for evaluation")
    
    # Load Bio_ClinicalBERT models
    logger.info("Loading Bio_ClinicalBERT models...")
    bio_models = find_latest_bio_clinicalbert_models()
    all_results = {}
    
    # Evaluate Bio_ClinicalBERT (Direct)
    if bio_models["direct"]:
        model, tokenizer, _ = bio_models["direct"]
        logger.info("Evaluating Bio_ClinicalBERT (Direct)...")
        results = evaluate_bio_clinicalbert_model(model, tokenizer, test_notes, gold_data)
        all_results["Bio_ClinicalBERT (Direct)"] = results
    else:
        logger.warning("No Bio_ClinicalBERT (Direct) model found, skipping.")
    
    # Evaluate Bio_ClinicalBERT (DSPy)
    if bio_models["dspy"]:
        model, tokenizer, _ = bio_models["dspy"]
        logger.info("Evaluating Bio_ClinicalBERT (DSPy)...")
        results = evaluate_bio_clinicalbert_model(model, tokenizer, test_notes, gold_data)
        all_results["Bio_ClinicalBERT (DSPy)"] = results
    else:
        logger.warning("No Bio_ClinicalBERT (DSPy) model found, skipping.")
    
    # Evaluate LLM and DSPy approaches
    logger.info("Evaluating LLM and DSPy approaches...")
    llm_dspy_results = evaluate_llm_and_dspy(test_notes, gold_data)
    all_results["Direct LLM"] = llm_dspy_results["direct"]
    all_results["DSPy"] = llm_dspy_results["dspy"]
    
    # Visualize and save
    results_dir = os.path.join("analysis", "comparison_results", 
                            datetime.now().strftime("%Y%m%d_%H%M%S") + "_bio_clinicalbert")
    visualize_results(all_results, results_dir)
    
    logger.info(f"Evaluation complete. Results saved to: {results_dir}")

if __name__ == "__main__":
    main() 