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
import argparse
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from utils.config import (
    GOLD_STANDARD_PATH, MAX_TEST_NOTES,
    BERT_OUTPUT_DIR, BERT_MODEL_NAME, BERT_MAX_LENGTH,
    STEP_2_GOLD_NER_DATA
)
from utils.data_transformation import load_from_jsonl
from utils.extraction import initialize_extractor
from utils.evaluation import calculate_entity_metrics
from utils.dataset import ADEDatasetProcessor
import evaluate  # <-- Add this import for seqeval

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_latest_bio_clinicalbert_models(direct_folder=None, dspy_folder=None):
    """Find and load the Bio_ClinicalBERT models for both approaches, optionally using user-specified folders."""
    models = {"direct": None, "dspy": None}

    # Direct approach
    if direct_folder:
        direct_path = os.path.join(BERT_OUTPUT_DIR, direct_folder)
        if os.path.exists(direct_path):
            try:
                if os.path.exists(os.path.join(direct_path, "pytorch_model.bin")) or \
                   os.path.exists(os.path.join(direct_path, "model.safetensors")):
                    model = AutoModelForTokenClassification.from_pretrained(direct_path)
                    tokenizer = AutoTokenizer.from_pretrained(direct_path)
                    models["direct"] = (model, tokenizer, direct_path)
                    logger.info(f"Loaded Bio_ClinicalBERT (direct) from user-specified: {direct_path}")
                else:
                    model_path = os.path.join(direct_path, "model")
                    if os.path.exists(model_path):
                        model = AutoModelForTokenClassification.from_pretrained(model_path)
                        tokenizer = AutoTokenizer.from_pretrained(model_path)
                        models["direct"] = (model, tokenizer, direct_path)
                        logger.info(f"Loaded Bio_ClinicalBERT (direct) from user-specified subdir: {model_path}")
                    else:
                        logger.error(f"Could not find model weights in {direct_path}")
            except Exception as e:
                logger.error(f"Error loading Bio_ClinicalBERT (direct) from user-specified: {e}")
        else:
            logger.error(f"User-specified direct model folder does not exist: {direct_path}")
    else:
        direct_models = glob.glob(os.path.join(BERT_OUTPUT_DIR, "direct_approach_*"))
        if direct_models:
            latest_direct = sorted(direct_models, key=os.path.getctime, reverse=True)[0]
            try:
                if os.path.exists(os.path.join(latest_direct, "pytorch_model.bin")) or \
                   os.path.exists(os.path.join(latest_direct, "model.safetensors")):
                    model = AutoModelForTokenClassification.from_pretrained(latest_direct)
                    tokenizer = AutoTokenizer.from_pretrained(latest_direct)
                    models["direct"] = (model, tokenizer, latest_direct)
                    logger.info(f"Loaded Bio_ClinicalBERT (direct) with weights from: {latest_direct}")
                else:
                    model_path = os.path.join(latest_direct, "model")
                    if os.path.exists(model_path):
                        model = AutoModelForTokenClassification.from_pretrained(model_path)
                        tokenizer = AutoTokenizer.from_pretrained(model_path)
                        models["direct"] = (model, tokenizer, latest_direct)
                        logger.info(f"Loaded Bio_ClinicalBERT (direct) from model subdirectory: {model_path}")
                    else:
                        logger.error(f"Could not find model weights in {latest_direct}")
            except Exception as e:
                logger.error(f"Error loading Bio_ClinicalBERT (direct): {e}")
        else:
            logger.warning("No Bio_ClinicalBERT direct models found")

    # DSPy approach
    if dspy_folder:
        dspy_path = os.path.join(BERT_OUTPUT_DIR, dspy_folder)
        if os.path.exists(dspy_path):
            try:
                if os.path.exists(os.path.join(dspy_path, "pytorch_model.bin")) or \
                   os.path.exists(os.path.join(dspy_path, "model.safetensors")):
                    model = AutoModelForTokenClassification.from_pretrained(dspy_path)
                    tokenizer = AutoTokenizer.from_pretrained(dspy_path)
                    models["dspy"] = (model, tokenizer, dspy_path)
                    logger.info(f"Loaded Bio_ClinicalBERT (dspy) from user-specified: {dspy_path}")
                else:
                    model_path = os.path.join(dspy_path, "model")
                    if os.path.exists(model_path):
                        model = AutoModelForTokenClassification.from_pretrained(model_path)
                        tokenizer = AutoTokenizer.from_pretrained(model_path)
                        models["dspy"] = (model, tokenizer, dspy_path)
                        logger.info(f"Loaded Bio_ClinicalBERT (dspy) from user-specified subdir: {model_path}")
                    else:
                        logger.error(f"Could not find model weights in {dspy_path}")
            except Exception as e:
                logger.error(f"Error loading Bio_ClinicalBERT (dspy) from user-specified: {e}")
        else:
            logger.error(f"User-specified dspy model folder does not exist: {dspy_path}")
    else:
        dspy_models = glob.glob(os.path.join(BERT_OUTPUT_DIR, "dspy_approach_*"))
        if dspy_models:
            latest_dspy = sorted(dspy_models, key=os.path.getctime, reverse=True)[0]
            try:
                if os.path.exists(os.path.join(latest_dspy, "pytorch_model.bin")) or \
                   os.path.exists(os.path.join(latest_dspy, "model.safetensors")):
                    model = AutoModelForTokenClassification.from_pretrained(latest_dspy)
                    tokenizer = AutoTokenizer.from_pretrained(latest_dspy)
                    models["dspy"] = (model, tokenizer, latest_dspy)
                    logger.info(f"Loaded Bio_ClinicalBERT (dspy) with weights from: {latest_dspy}")
                else:
                    model_path = os.path.join(latest_dspy, "model")
                    if os.path.exists(model_path):
                        model = AutoModelForTokenClassification.from_pretrained(model_path)
                        tokenizer = AutoTokenizer.from_pretrained(model_path)
                        models["dspy"] = (model, tokenizer, latest_dspy)
                        logger.info(f"Loaded Bio_ClinicalBERT (dspy) from model subdirectory: {model_path}")
                    else:
                        logger.error(f"Could not find model weights in {latest_dspy}")
            except Exception as e:
                logger.error(f"Error loading Bio_ClinicalBERT (dspy): {e}")
        else:
            logger.warning("No Bio_ClinicalBERT dspy models found")
    return models

def evaluate_bio_clinicalbert_model(model, tokenizer, test_texts, gold_data):
    """Evaluate a Bio_ClinicalBERT model on the gold standard using token-level BIO evaluation (seqeval)."""
    seqeval = evaluate.load("seqeval")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model.to(device)
    model.eval()
    
    all_pred_tags = []
    all_gold_tags = []
    
    for idx, gold in enumerate(tqdm(gold_data, desc="Evaluating Bio_ClinicalBERT (BIO eval)")):
        text = gold["text"]
        
        # Use consistent tokenization for both prediction and gold alignment
        # First tokenize for model input (with special tokens and padding)
        model_inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=BERT_MAX_LENGTH, 
            padding="max_length"
        )
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        
        # Also tokenize for alignment (without special tokens, with offset mapping)
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
        
        # Extract predictions for actual tokens (skip [CLS] and [SEP])
        # predictions[0] is [CLS], predictions[1:-1] are actual tokens, predictions[-1] is [SEP]
        actual_predictions = predictions[1:len(tokens)+1]  # Skip [CLS], take only actual token predictions
        
        # Convert predictions to labels
        pred_tags = []
        for pred in actual_predictions:
            label = model.config.id2label.get(int(pred), 'O')
            pred_tags.append(label)
        
        # Generate gold tags using the same tokenization
        gold_entities = gold.get('entities', [])
        gold_tags = ['O'] * len(tokens)
        
        # Assign BIO tags based on entity spans using the same offset mapping
        for entity in gold_entities:
            entity_start = entity.get('start')
            entity_end = entity.get('end')
            entity_label = entity.get('label')
            
            # Find tokens that overlap with this entity
            for i, (start, end) in enumerate(offset_mapping):
                # Skip if token is outside entity boundaries
                if end <= entity_start or start >= entity_end:
                    continue
                
                # First token of entity gets B- prefix
                if i == 0 or offset_mapping[i-1][1] <= entity_start:
                    gold_tags[i] = f'B-{entity_label}'
                else:
                    gold_tags[i] = f'I-{entity_label}'
        
        # Ensure both sequences have the same length
        min_len = min(len(pred_tags), len(gold_tags))
        pred_tags = pred_tags[:min_len]
        gold_tags = gold_tags[:min_len]
        
        all_pred_tags.append(pred_tags)
        all_gold_tags.append(gold_tags)
        
        if idx < 3:
            logger.info(f"Example {idx+1}:")
            logger.info(f"Text: {text[:100]}...")
            logger.info(f"Tokens: {tokens[:min_len]}")
            logger.info(f"Pred tags: {pred_tags}")
            logger.info(f"Gold tags: {gold_tags}")
            logger.info(f"Entities in gold: {gold_entities}")
    
    # Compute metrics
    results = seqeval.compute(predictions=all_pred_tags, references=all_gold_tags)
    logger.info(f"\n========== BERT MODEL SEQEVAL EVALUATION SUMMARY ==========")
    logger.info(f"F1: {results.get('overall_f1', 0):.4f}, Precision: {results.get('overall_precision', 0):.4f}, Recall: {results.get('overall_recall', 0):.4f}")
    logger.info(f"================================================\n")
    
    # Return a list of dicts for compatibility
    return [{
        'overall': {
            'precision': results.get('overall_precision', 0),
            'recall': results.get('overall_recall', 0),
            'f1': results.get('overall_f1', 0),
        }
    } for _ in range(len(all_pred_tags))]

def spans_to_bio_tags(text, entities, tokenizer):
    """Convert entity spans to BIO tag sequence for a given text using the tokenizer."""
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(encoding.input_ids)
    offset_mapping = encoding.offset_mapping
    tags = ['O'] * len(tokens)
    for entity in entities:
        entity_start = entity.get('start')
        entity_end = entity.get('end')
        entity_label = entity.get('label')
        for i, (start, end) in enumerate(offset_mapping):
            if end <= entity_start or start >= entity_end:
                continue
            if i == 0 or offset_mapping[i-1][1] <= entity_start:
                tags[i] = f'B-{entity_label}'
            else:
                tags[i] = f'I-{entity_label}'
    return tokens, tags


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
    parser = argparse.ArgumentParser(description="Comprehensive Bio_ClinicalBERT ADE evaluation")
    parser.add_argument('--overwrite-llm', action='store_true', help='Force rerun of LLM and DSPy extraction (costs money)')
    parser.add_argument('--direct-bert-folder', type=str, default=None, help='Specify a subfolder in bert_ade_extractor/ to use for the direct approach (e.g., direct_approach_20250528_010125). If not provided, uses the latest.')
    parser.add_argument('--dspy-bert-folder', type=str, default=None, help='Specify a subfolder in bert_ade_extractor/ to use for the dspy approach (e.g., dspy_approach_20250528_010125). If not provided, uses the latest.')
    args = parser.parse_args()

    # Load gold standard data - use NER format
    logger.info(f"Loading gold standard NER data from {STEP_2_GOLD_NER_DATA}")
    gold_data = load_from_jsonl(STEP_2_GOLD_NER_DATA)
    gold_data = gold_data[:MAX_TEST_NOTES]
    test_notes = [entry["text"] for entry in gold_data if "text" in entry]
    logger.info(f"Using {len(test_notes)} gold standard notes for evaluation")

    # Prepare analysis directory
    results_dir = os.path.join("analysis", "comparison_results", \
                            datetime.now().strftime("%Y%m%d_%H%M%S") + "_bio_clinicalbert")
    os.makedirs(results_dir, exist_ok=True)

    # Prepare persistent LLM cache directory
    llm_cache_dir = os.path.join("analysis", "llm_cache")
    os.makedirs(llm_cache_dir, exist_ok=True)
    llm_direct_path = os.path.join(llm_cache_dir, "llm_direct.jsonl")
    llm_dspy_path = os.path.join(llm_cache_dir, "llm_dspy.jsonl")
    llm_direct_exists = os.path.exists(llm_direct_path)
    llm_dspy_exists = os.path.exists(llm_dspy_path)

    # Load Bio_ClinicalBERT models
    logger.info("Loading Bio_ClinicalBERT models...")
    bio_models = find_latest_bio_clinicalbert_models(
        direct_folder=args.direct_bert_folder,
        dspy_folder=args.dspy_bert_folder
    )
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

    # LLM and DSPy extraction caching (persistent across runs)
    logger.info("Evaluating LLM and DSPy approaches...")
    if not args.overwrite_llm and llm_direct_exists and llm_dspy_exists:
        logger.info("Loading cached LLM and DSPy extraction results from analysis/llm_cache/ ...")
        with open(llm_direct_path, 'r') as f:
            direct_ner = [json.loads(line) for line in f]
        with open(llm_dspy_path, 'r') as f:
            dspy_ner = [json.loads(line) for line in f]
    else:
        logger.info("Extracting ADEs using Direct approach (pipeline logic)...")
        direct_extractor = initialize_extractor(mode="direct")
        direct_processor = ADEDatasetProcessor(extractor=direct_extractor)
        direct_extracted = direct_processor.extract_ades_batched(test_notes)
        direct_ner = direct_processor.prepare_ner_data(direct_extracted)
        # Save
        with open(llm_direct_path, 'w') as f:
            for rec in direct_ner:
                f.write(json.dumps(rec) + '\n')

        logger.info("Extracting ADEs using DSPy approach (pipeline logic)...")
        dspy_extractor = initialize_extractor(mode="dspy")
        dspy_processor = ADEDatasetProcessor(extractor=dspy_extractor)
        dspy_extracted = dspy_processor.extract_ades_batched(test_notes)
        dspy_ner = dspy_processor.prepare_ner_data(dspy_extracted)
        # Save
        with open(llm_dspy_path, 'w') as f:
            for rec in dspy_ner:
                f.write(json.dumps(rec) + '\n')

    # Evaluate LLM and DSPy approaches
    results = {"direct": [], "dspy": []}
    for approach, ner_data in zip(["direct", "dspy"], [direct_ner, dspy_ner]):
        for idx, (record, gold) in enumerate(zip(ner_data, gold_data)):
            # Convert entities to drugs/adverse_events format for metric calculation
            drugs = []
            adverse_events = []
            text = record["text"]
            for entity in record["entities"]:
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
            metrics = calculate_entity_metrics(pred_record_converted, gold_converted)
            results[approach].append(metrics)
    all_results["Direct LLM"] = results["direct"]
    all_results["DSPy"] = results["dspy"]

    # Visualize and save
    visualize_results(all_results, results_dir)
    logger.info(f"Evaluation complete. Results saved to: {results_dir}")

if __name__ == "__main__":
    main() 