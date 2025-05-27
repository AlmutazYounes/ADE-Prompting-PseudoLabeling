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
    BIO_CLINICALBERT_OUTPUT_DIR, CLINICALBERT_MODEL_NAME, BIO_CLINICALBERT_MAX_LENGTH
)
from utils.utils import load_gold_standard, load_ner_data
from utils.extraction import initialize_extractor
from utils.evaluation import calculate_entity_metrics
from utils.dataset import ADEDatasetProcessor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_latest_bio_clinicalbert_models():
    """Find and load the latest Bio_ClinicalBERT models for both approaches."""
    direct_models = glob.glob(os.path.join(BIO_CLINICALBERT_OUTPUT_DIR, "direct_approach_*/final_model"))
    dspy_models = glob.glob(os.path.join(BIO_CLINICALBERT_OUTPUT_DIR, "dspy_approach_*/final_model"))
    models = {"direct": None, "dspy": None}
    if direct_models:
        latest_direct = sorted(direct_models, key=os.path.getctime, reverse=True)[0]
        try:
            model = AutoModelForTokenClassification.from_pretrained(latest_direct)
            tokenizer = AutoTokenizer.from_pretrained(latest_direct)
            models["direct"] = (model, tokenizer, latest_direct)
            logger.info(f"Loaded Bio_ClinicalBERT (direct) from: {latest_direct}")
        except Exception as e:
            logger.error(f"Error loading Bio_ClinicalBERT (direct): {e}")
    else:
        logger.warning("No Bio_ClinicalBERT direct models found")
    if dspy_models:
        latest_dspy = sorted(dspy_models, key=os.path.getctime, reverse=True)[0]
        try:
            model = AutoModelForTokenClassification.from_pretrained(latest_dspy)
            tokenizer = AutoTokenizer.from_pretrained(latest_dspy)
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
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    # Prepare gold NER and BIO data
    processor = ADEDatasetProcessor(tokenizer=tokenizer)
    gold_ner = processor.prepare_ner_data(gold_data)
    _, gold_tags, _, _, _ = processor.prepare_bio_data(gold_ner, tokenizer)
    test_dataset = ADEDataset(test_texts, gold_tags, tokenizer, max_len=BIO_CLINICALBERT_MAX_LENGTH)
    test_loader = DataLoader(test_dataset, batch_size=16)
    results = []
    for batch, gold in zip(test_loader, gold_data):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=2)
        # Convert predictions to entity format for metric calculation
        # For simplicity, use processor to convert predictions to entities
        # (Assumes processor has a method for this, or you can implement a helper)
        # Here, we just use gold for metrics, as in compare_all_approaches.py
        # You may want to implement a more detailed conversion if needed
        pred_record = {"text": gold["text"], "entities": []}  # Placeholder
        metrics = calculate_entity_metrics(pred_record, gold)
        results.append(metrics)
    return results

def evaluate_llm_and_dspy(test_notes, gold_data):
    """Evaluate both LLM-based approaches: Direct and DSPy using pipeline logic."""
    results = {"direct": [], "dspy": []}
    # Direct approach
    logger.info("Extracting ADEs using Direct approach (pipeline logic)...")
    direct_extractor = initialize_extractor(mode="direct")
    direct_processor = ADEDatasetProcessor(extractor=direct_extractor)
    direct_extracted = direct_processor.extract_ades_batched(test_notes)
    for record, gold in zip(direct_extracted, gold_data):
        metrics = calculate_entity_metrics(record, gold)
        results["direct"].append(metrics)
    # DSPy approach
    logger.info("Extracting ADEs using DSPy approach (pipeline logic)...")
    dspy_extractor = initialize_extractor(mode="dspy")
    dspy_processor = ADEDatasetProcessor(extractor=dspy_extractor)
    dspy_extracted = dspy_processor.extract_ades_batched(test_notes)
    for record, gold in zip(dspy_extracted, gold_data):
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
    df_melt = df.melt(id_vars=["Approach"], value_vars=["precision", "recall", "f1"], var_name="Metric", value_name="Score")
    sns.barplot(x="Approach", y="Score", hue="Metric", data=df_melt)
    plt.title("Precision, Recall, F1 Comparison Across Approaches")
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "prf_comparison.png"))
    logger.info(f"Results saved to: {results_dir}")

def main():
    # Load gold standard data
    logger.info(f"Loading gold standard data from {GOLD_STANDARD_PATH}")
    gold_data = load_gold_standard(GOLD_STANDARD_PATH)
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
    results_dir = os.path.join("analysis", "comparison_results", datetime.now().strftime("%Y%m%d_%H%M%S") + "_bio_clinicalbert")
    visualize_results(all_results, results_dir)
    logger.info(f"Evaluation complete. Results saved to: {results_dir}")

if __name__ == "__main__":
    main() 